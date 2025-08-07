import csv
import re
import json
import asyncio
from itertools import islice
from pathlib import Path

import numpy as np
from collections import Counter, defaultdict
from typing import Union, List, Dict, Optional, Any, Tuple, Set
import difflib
import time

from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    extract_inner_content,
    normalize_sentence,
    fuzzy_lookup_fname, gen_chunk_node_id, gen_entity_node_id, extract_clause, extract_json_from_response,
    split_internal_external,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,

)

from .config import GLOBAL_CONCURRENT_LLM_LIMIT, LOCAL_LLM_PRODUCE, GLOBAL_LLM_PRODUCE, ENABLE_TRIPLE_EXTRACTION, \
    PREPROCESS_QUERY_WITH_HYDE
from .prompt import GRAPH_FIELD_SEP, PROMPTS, ENTITY_TYPES



# 文本分割，默认使用token size分割，也可以使用自定义的分隔符分割
# 实体提取
# 社区报告生成
# 不同查询模式（本地，全局，朴素）


# 压缩描述（实体或关系）
async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func = global_config["cheap_model_func"]  # 使用低成本模型
    llm_max_tokens = global_config["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    description = clean_description(description)

    # tokenize描述
    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:
        return description  # 无需摘要则直接返回原始描述

    # 构造摘要提示词
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
        name=entity_or_relation_name,
    )
    use_prompt = prompt_template.format(**context_base)

    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)

    return summary  # 返回压缩后的摘要内容



# 单条实体解析器
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    args: list[str]
) -> Dict | None:
    """解析 ("entity"|...) 元组 → 规范化并过滤非法类型"""
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None  # 非法格式丢弃

    entity_name = canonical_name(record_attributes[1])        # 标准化名称
    entity_type_raw = record_attributes[2]                    # 原始类型
    entity_type = normalize_entity_type(entity_type_raw)      # 类型归一化
    entity_description = clean_description(clean_str(record_attributes[3]))      # 描述清洗

    entity_file_Name = args[0]
    entity_regulation = args[1]

    # 丢弃无效实体
    if not entity_name or entity_type is None:
        logger.debug(f"🚫 丢弃无效实体: {entity_name=} {entity_type_raw=}")
        return None

    return {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "description": entity_description,
        "source_id": chunk_key,  # 来源chunk
        "name": entity_name,
        "file_name": entity_file_Name,
        "regulation": entity_regulation,
    }



# 解析单条关系抽取元组
async def _handle_single_relationship_extraction(
    record_attributes: list[str],  # 元组内容，例如 ("relationship", "张三", "李四", "合作关系", "战略合作", 1.0)
    chunk_key: str,                # 当前文本块的ID
    global_config: dict,          # 配置项（用于压缩或归一化）
) -> Dict | None:
    # 至少包含6个字段，且开头标记为 "relationship"
    if len(record_attributes) < 6 or record_attributes[0] != '"relationship"':
        return None

    # 规范化实体名
    source_entity = canonical_name(record_attributes[1])
    target_entity = canonical_name(record_attributes[2])

    # 清洗描述
    description = clean_description(clean_str(record_attributes[3]))

    # 归一化关系类型
    relation_type_raw = record_attributes[4]
    relation_type = normalize_relation_type(relation_type_raw)

    # 尝试解析权重（若失败则默认1.0）
    try:
        weight = float(record_attributes[5])
    except ValueError:
        weight = 1.0

    # 若缺少源或目标实体，丢弃
    if not source_entity or not target_entity:
        return None

    # 返回结构化关系数据
    return {
        "source": source_entity,
        "target": target_entity,
        "description": description,
        "weight": weight,
        "source_id": chunk_key,
        "relation_type": relation_type,
    }


# 合并同名实体节点并写入图谱
# ================== graph_ops.py ==================
async def _merge_nodes_then_upsert(
    node_id: str,
    nodes_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    · node_id 在块内唯一
    · 仅同 ID 条目才会进入合并
    · 块节点( entity_type == "规则条目_块" ) 才保留 labels
    """
    already_node = await knwoledge_graph_inst.get_node(node_id)

    collected_types        = [dp["entity_type"] for dp in nodes_data]
    collected_descs        = [dp["description"]  for dp in nodes_data]
    collected_source       = [dp["source_id"]    for dp in nodes_data]
    collected_file_names   = [dp.get("file_name", "unknown")  for dp in nodes_data]
    collected_regulations  = [dp.get("regulation", "unknown") for dp in nodes_data]

    # 🆕 仅暂存 labels，稍后判断是否写回
    collected_labels: list[str] = []
    for dp in nodes_data:
        if "labels" in dp and dp["labels"]:
            # 允许 list / str 两种形式
            if isinstance(dp["labels"], list):
                collected_labels.extend(dp["labels"])
            else:
                collected_labels.extend(
                    [lbl.strip() for lbl in str(dp["labels"]).split(",") if lbl.strip()]
                )

    if already_node:
        collected_types.append(already_node["entity_type"])
        collected_descs.append(already_node["description"])
        collected_source.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        collected_file_names.append(already_node.get("file_name", "unknown"))
        collected_regulations.append(already_node.get("regulation", "unknown"))

        # 已有节点若存 labels，也并进来（仅块才会有）
        if "labels" in already_node and already_node["labels"]:
            if isinstance(already_node["labels"], list):
                collected_labels.extend(already_node["labels"])
            else:
                collected_labels.extend(
                    [lbl.strip() for lbl in str(already_node["labels"]).split(",") if lbl.strip()]
                )

    # ======== 汇总字段 ========
    entity_type = max(Counter(collected_types), key=Counter(collected_types).get)
    description = GRAPH_FIELD_SEP.join(sorted(set(collected_descs)))
    source_id   = GRAPH_FIELD_SEP.join(sorted(set(collected_source)))
    file_name   = GRAPH_FIELD_SEP.join(sorted(set(collected_file_names)))
    regulation  = GRAPH_FIELD_SEP.join(sorted(set(collected_regulations)))
    labels_uniq = sorted(set(collected_labels))

    description = await _handle_entity_relation_summary(node_id, description, global_config)

    # original_name 只用于显示
    original_name = next(
        (dp.get("original_name") for dp in nodes_data if dp.get("original_name")),
        node_id,
    )

    node_data = {
        "entity_type": entity_type,
        "description": description,
        "source_id":   source_id,
        "file_name":   file_name,
        "regulation":  regulation,
        "name":        original_name,
    }

    # 👉 只有块节点才写 labels
    if entity_type == "规则条目_块" and labels_uniq:
        node_data["labels"] = labels_uniq

    await knwoledge_graph_inst.upsert_node(node_id, node_data=node_data)

    node_data["entity_name"] = node_id
    return node_data

# 合并相同实体对之间的边（无向）
# ================== graph_ops.py ==================
async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knwoledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    """
    * 不再 canonical_name；ID 已经全局唯一；
    * 仍然将边视作无向，按排序后的 key 聚合。
    """
    key = tuple(sorted((src_id, tgt_id)))
    already_edge = await knwoledge_graph_inst.get_edge(*key) if await knwoledge_graph_inst.has_edge(*key) else None

    agg_weight = sum(ed.get("weight", 1.0) for ed in edges_data)
    agg_desc   = [ed["description"] for ed in edges_data]
    agg_type   = [ed.get("relation_type", "RELATED") for ed in edges_data]

    if already_edge:
        agg_weight += already_edge.get("weight", 0.0)
        agg_desc.append(already_edge.get("description", ""))
        agg_type.append(already_edge.get("relation_type", "RELATED"))

    description    = GRAPH_FIELD_SEP.join(sorted(set(agg_desc)))
    relation_type  = Counter(agg_type).most_common(1)[0][0]
    description    = await _handle_entity_relation_summary(key, description, global_config)

    await knwoledge_graph_inst.upsert_edge(
        *key,
        edge_data=dict(
            weight=agg_weight,
            description=description,
            source_id=GRAPH_FIELD_SEP.join({ed["source_id"] for ed in edges_data}),
            relation_type=relation_type,
        ),
    )


from tqdm.asyncio import tqdm_asyncio

# 实体抽取与关系提取的核心函数
async def extract_entities(
    chunks: dict[str, TextChunkSchema],  # 文本块，键为chunk_id，值为文本内容和元数据
    knwoledge_graph_inst: BaseGraphStorage,  # 图数据库实例，用于存储抽取后的实体和关系
    entity_vdb: BaseVectorStorage,  # 实体向量库（用于存储实体向量）
    global_config: dict,  # 全局配置，包含模型函数、Token限制等
    args: list[str],
    using_amazon_bedrock: bool=False,  # 是否使用Amazon Bedrock（影响历史消息格式）
) -> Union[BaseGraphStorage, None]:

    # 从配置中获取模型函数和最大多轮抽取次数
    use_llm_func: callable = global_config["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]

    # 将chunks转成列表，方便遍历处理
    ordered_chunks = list(chunks.items())




    # 构造基本提示词上下文
    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        relationship_types=",".join(PROMPTS["DEFAULT_RELATIONSHIP_TYPES"]),
    )

    entity_extract_prompt_2nd = PROMPTS["entity_and_relationship_discrimination"]
    context_base_2nd = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        relationship_types=",".join(PROMPTS["DEFAULT_RELATIONSHIP_TYPES"]),
    )

    entity_extract_prompt_3rd = PROMPTS["final_judgment_on_entities_and_relationships"]
    context_base_3rd = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
        relationship_types=",".join(PROMPTS["DEFAULT_RELATIONSHIP_TYPES"]),
    )


    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    # 抽取过程的计数器
    already_processed = 0
    already_entities = 0
    already_relations = 0

    # 单个 chunk 的处理逻辑
    # ================== extract_entities.py ==================
    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        """
        * 先把【块】本身当节点插入；
        * 同块实体用 gen_entity_node_id 去重，跨块不去重；
        * 每个实体 ➜ 块连一条 IN_CHUNK 边；
        * 实体–实体边仅限本块。
        """
        nonlocal already_processed, already_entities, already_relations

        chunk_id: str = chunk_key_dp[0]  # ⇢ 原来的 key
        chunk_dp: TextChunkSchema = chunk_key_dp[1]
        content: str = chunk_dp["content"]

        meta = chunk_dp.get("meta", {})
        file_name = meta.get("file_name", "unknown")
        regulation = meta.get("regulation", "unknown")

        label_prompt = PROMPTS["select_labels_by_rule"].format(labels=",".join(PROMPTS["DEFAULT_LABELS"]), rule_text=content)
        label_res = await use_llm_func(label_prompt)

        cleaned_labels = label_res.strip().removeprefix("```json").removesuffix("```").strip()

        cleaned_labels = re.sub(r"<think>[\s\S]*?</think>", "", cleaned_labels)

        try:
            labels = json.loads(cleaned_labels)
        except Exception as e:
            print(f"标签反序列化失败: {e}")
            print(f"原始内容: {label_res}")
            labels = []



        # ---- ① 先创建块节点 ----------------------------------
        chunk_node_id = gen_chunk_node_id(chunk_id)
        chunk_node_data = {
            "entity_type": "规则条目_块",
            "description": content,
            "source_id": chunk_id,
            "file_name": file_name,
            "regulation": regulation,
            "name": f"Chunk-{chunk_id}",
            "labels":labels,
        }



        maybe_nodes = defaultdict(list)
        maybe_nodes[chunk_node_id].append(chunk_node_data)

        # ---- ② 正常走 LLM 抽取 -------------------------------
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)
        if isinstance(final_result, list):
            final_result = final_result[0]["text"]

        # 对话式多轮抽取
        # history = pack_user_ass_to_openai_messages(hint_prompt, final_result, using_amazon_bedrock)
        # for now_glean_index in range(entity_extract_max_gleaning):
        #     glean_result = await use_llm_func(continue_prompt, history_messages=history)
        #     history += pack_user_ass_to_openai_messages(continue_prompt, glean_result, using_amazon_bedrock)
        #     final_result += glean_result
        #     if now_glean_index == entity_extract_max_gleaning - 1:
        #         break
        #     if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
        #     if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
        #         break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        # 👉 记录「原始实体名 → 本块唯一 ID」
        entname2id: dict[str, str] = {}
        maybe_edges = defaultdict(list)

        for record in records:
            m = re.search(r"\((.*)\)", record)
            if m is None:
                continue
            record_attributes = split_string_by_multi_markers(m.group(1), [context_base["tuple_delimiter"]])

            # ---------- 实体 ----------
            ent = await _handle_single_entity_extraction(record_attributes, chunk_id, args)
            if ent:
                raw_name = ent["entity_name"]
                ent_id = gen_entity_node_id(chunk_id, raw_name)

                # 同块内去重：已存在就跳过
                if ent_id in maybe_nodes:
                    continue

                ent["original_name"] = raw_name  # 备用
                ent["entity_name"] = ent_id  # 👈 后续流程把它当 node_id 用
                maybe_nodes[ent_id].append(ent)
                entname2id[raw_name] = ent_id

                # ➜ 块连边
                maybe_edges[tuple(sorted((ent_id, chunk_node_id)))].append(
                    dict(
                        weight=0.0,
                        description="块节点包含该实体",
                        source_id=chunk_id,
                        relation_type="包含实体",
                    )
                )
                continue

            # ---------- 关系 ----------
            rel = await _handle_single_relationship_extraction(record_attributes, chunk_id, global_config)
            if rel:
                src_raw, tgt_raw = rel["source"], rel["target"]
                if src_raw not in entname2id or tgt_raw not in entname2id:
                    # 😂 LLM 胡说八道的边，里面的实体根本没出现
                    continue
                src_id, tgt_id = entname2id[src_raw], entname2id[tgt_raw]
                rel["source"], rel["target"] = src_id, tgt_id
                maybe_edges[tuple(sorted((src_id, tgt_id)))].append(rel)

        # ---- ③ 更新统计 & 进度条 -------------------------------
        already_processed += 1
        already_entities += len(maybe_nodes) - 1  # 不计块节点
        already_relations += len(maybe_edges)

        now_ticks = PROMPTS["process_tickers"][already_processed % len(PROMPTS["process_tickers"])]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed * 100 // len(ordered_chunks)}%) chunks,  "
            f"{already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )



        return dict(maybe_nodes), dict(maybe_edges)

    #========================== 函数到这里 =====================================

    # 并发处理所有 chunks，并用 tqdm 展示抽取进度条
    results = await tqdm_asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks],
        desc="🧠 Entity Extraction",
    )

    print()  # 抽取完成后换行，清理进度条

    logger.info("合并所有节点/边开始")

    # 整合所有结果：合并所有节点和边
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            maybe_edges[tuple(sorted(k))].extend(v)  # 无向图需排序


    logger.info("同名实体插入数据库开始")

    # 合并同名实体并插入数据库
    all_entities_data = await asyncio.gather(
        *[_merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config) for k, v in maybe_nodes.items()]
    )

    logger.info("同名边插入数据库开始")
    # 合并边数据并插入数据库
    await asyncio.gather(
        *[_merge_edges_then_upsert(k[0], k[1], v, knwoledge_graph_inst, global_config) for k, v in maybe_edges.items()]
    )

    # 若无实体，打印警告
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None

    logger.info("插入向量库开始")
    # 若存在向量库，则构建向量插入
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + ":" + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)


    return knwoledge_graph_inst  # 返回更新后的图数据库实例



# ================================================================

import csv
import asyncio
from pathlib import Path
from itertools import islice
from typing import Set, Tuple, Dict, List

import csv
import asyncio
from pathlib import Path
from itertools import islice
from typing import Set, Tuple, Dict, List
import json

async def link_rules(
    knwoledge_graph_inst,
    community_reports,
    inner_regulation_vdb,
    outer_regulation_vdb,
    global_config: dict,
    *,
    batch_llm: int = 256,
    label_weight: float = 0.5,
    entity_weight: float = 0.5,
    jaccard_cutoff: float = 0.2,
    entity_cutoff: float = 0.2,
    relation: str = "OR",
    embed_top_k: int = 20,
    embed_sim_cutoff: float = 0.5,
    embed_weight: float = 0.5,
    combined_cutoff: float = 0.6,
    llm_threshold: float = -1.0,
) -> int:
    """
    输出 3 份 CSV：
      1) link_rules_params.csv
      2) jaccard_pairs.csv
      3) embed_pairs.csv

    初筛（first screening） = Jaccard+Embedding 融合候选数
    复筛（second screening） = LLM 评估通过并写入数
    """
    BASE_DIR = Path("/home/weida/PycharmProjects/laip_graphrag/中间数据")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # ——— A. 写入参数 CSV ———
    params_csv = BASE_DIR / "link_rules_params.csv"
    with params_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["param", "value"])
        for name, val in [
            ("batch_llm", batch_llm),
            ("label_weight", label_weight),
            ("entity_weight", entity_weight),
            ("jaccard_cutoff", jaccard_cutoff),
            ("entity_cutoff", entity_cutoff),
            ("relation", relation),
            ("embed_top_k", embed_top_k),
            ("embed_sim_cutoff", embed_sim_cutoff),
            ("embed_weight", embed_weight),
            ("combined_cutoff", combined_cutoff),
            ("llm_threshold", llm_threshold),
        ]:
            w.writerow([name, val])
    print(f"[INFO] 已写参数 {params_csv}")

    # ——— 0A. Jaccard 通道 ———
    jaccard_weight = 1.0 - embed_weight
    jaccard_results = await knwoledge_graph_inst.fetch_candidate_chunk_pairs(
        label_weight=label_weight,
        entity_weight=entity_weight,
        label_cutoff=jaccard_cutoff,
        entity_cutoff=entity_cutoff,
        relation=relation,
    )
    pre_scores: Dict[Tuple[str, str], float] = {
        (iid, oid): jaccard_weight * score
        for iid, oid, score in jaccard_results
    }
    jaccard_pairs: Set[Tuple[str, str]] = set(pre_scores.keys())
    print(f"🎯 Jaccard 通道命中 {len(jaccard_pairs)} 对，示例：{list(pre_scores.items())[:3]}")

    # ——— 0B. Embedding 通道 ———
    all_blocks = await knwoledge_graph_inst.fetch_all_rule_blocks()
    inner_ids = {b["id"] for b in all_blocks if b.get("regulation") and "内规" in b["regulation"]}
    outer_ids = {b["id"] for b in all_blocks if b.get("regulation") and "内规" not in b["regulation"]}
    desc_map = await knwoledge_graph_inst.fetch_chunk_descriptions(list(inner_ids | outer_ids))

    embed_only_count = 0
    embed_pairs_all: Set[Tuple[str, str]] = set()

    async def enrich_by_embed(query_ids: Set[str], target_vdb, target_set: Set[str], reverse=False):
        nonlocal embed_only_count
        for qid in query_ids:
            q_desc = desc_map.get(qid, {}).get("desc", "")
            if not q_desc:
                continue
            hits = await target_vdb.query_regulation(q_desc, top_k=embed_top_k)
            for rank, hit in enumerate(hits):
                hid = hit.get("regulation_id")
                certainty = hit.get("certainty", 0.0)
                if not hid or certainty < embed_sim_cutoff or hid not in target_set:
                    continue
                pair = (qid, hid) if not reverse else (hid, qid)
                embed_pairs_all.add(pair)
                if pair not in pre_scores:
                    embed_only_count += 1
                    pre_scores[pair] = 0.0
                score_add = embed_weight * certainty * (embed_top_k - rank) / embed_top_k
                pre_scores[pair] = max(pre_scores[pair] + score_add, pre_scores[pair])

    await enrich_by_embed(inner_ids, outer_regulation_vdb, outer_ids, reverse=False)
    await enrich_by_embed(outer_ids, inner_regulation_vdb, inner_ids, reverse=True)
    print(f"🔎 嵌入通道独有命中 {embed_only_count} 对，总命中 {len(embed_pairs_all)} 对")

    # 写入 Jaccard & Embedding CSV
    def write_pairs_csv(pairs: Set[Tuple[str, str]], filename: str):
        path = BASE_DIR / filename
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["internal", "external"])
            for iid, oid in pairs:
                _, _, i_data, o_data = split_internal_external(iid, oid, desc_map)
                w.writerow([
                    extract_clause(i_data.get("desc", "")),
                    extract_clause(o_data.get("desc", "")),
                ])
        print(f"[INFO] 已写 {filename}")

    write_pairs_csv(jaccard_pairs, "jaccard_pairs.csv")
    write_pairs_csv(embed_pairs_all, "embed_pairs.csv")

    # ——— 0C. 融合过滤（初筛） ———
    refined_pairs = [p for p, s in pre_scores.items() if s >= combined_cutoff]
    first_screen_count = len(refined_pairs)
    print(f"🏷 初筛（Jaccard+Embedding 融合）候选 {first_screen_count} 对")

    # —— 复筛准备 ——
    it = iter(refined_pairs)
    batch_num = 0
    written = 0

    llm_func = global_config["best_model_func"]
    prompt_tpl = PROMPTS["regulation_consistency_evaluation"]

    # 社区报告预取
    comm_ids = {d.get("community_hashid") or d.get("community_id") for d in desc_map.values() if d}
    comm_ids.discard(None)
    comm_raw = await community_reports.get_by_ids(list(comm_ids))
    comm_map = {
        cid: (r.get("report_string") if isinstance(r, dict) else "")
        for cid, r in zip(comm_ids, comm_raw)
    }

    # ——— 1. 复筛（LLM 评估） ———
    while True:
        batch = list(islice(it, batch_llm))
        if not batch:
            break
        batch_num += 1

        llm_tasks = []
        for iid, oid in batch:
            _, _, i_data, o_data = split_internal_external(iid, oid, desc_map)
            prompt = prompt_tpl.format(
                query=i_data.get("desc", ""),
                response=o_data.get("desc", ""),
                query_community_report=comm_map.get(i_data.get("community_hashid") or i_data.get("community_id"), ""),
                response_community_report=comm_map.get(o_data.get("community_hashid") or o_data.get("community_id"), ""),
            )
            llm_tasks.append(llm_func(prompt))

        llm_results = await asyncio.gather(*llm_tasks)
        for (iid, oid), rsp in zip(batch, llm_results):
            text = rsp[0].get("text", "") if isinstance(rsp, list) else rsp
            try:
                data = extract_json_from_response(text)
                score = float(data["score"])
                explanation = data["explanation"]
            except Exception:
                continue
            if score < llm_threshold:
                continue

            internal_id, external_id, *_ = split_internal_external(iid, oid, desc_map)
            await knwoledge_graph_inst.upsert_match_with_edge(
                internal_id,
                external_id,
                score,
                json.dumps(explanation, ensure_ascii=False)
            )
            written += 1

        print(f"[LLM] Batch {batch_num} 完成，累计通过写入 {written} 条")

    # ——— 结束日志 ——
    print(f"🔍 复筛（LLM 评估）通过并写入：{written} 条")
    print(f"✅ 全流程结束：初筛候选 {first_screen_count} → 复筛写入 {written}")
    return written




# async def link_rules_backup(
#     knwoledge_graph_inst,
#     community_reports,
#     inner_regulation_vdb,
#     outer_regulation_vdb,
#     global_config: dict,
#     *,
#     batch_llm: int = 256,
#     label_weight: float = 0.5,
#     entity_weight: float = 0.5,
#     jaccard_cutoff: float = 0.05,
#     entity_cutoff: float = 0.015,
#     relation: str = "AND",
#
#     embed_top_k: int = 20,
#     embed_sim_cutoff: float = 0.5,
#
#     embed_weight: float = 0.7,
#
#     combined_cutoff: float = 0.6,
#
#     llm_threshold: float = 6.0,
#
# ) -> int:
#     jaccard_weight = 1.0 - embed_weight
#
#     # ---------- 0A. Jaccard ----------
#     jaccard_pairs = await knwoledge_graph_inst.fetch_candidate_chunk_pairs(
#         label_weight=label_weight,
#         entity_weight=entity_weight,
#         label_cutoff=jaccard_cutoff,
#         entity_cutoff=entity_cutoff,
#         relation = relation,
#     )
#     pre_scores: Dict[Tuple[str, str], float] = {
#         p: jaccard_weight for p in jaccard_pairs
#     }
#     print(f"🎯 Jaccard 命中 {len(pre_scores)} 对")
#
#     # ---------- 0B. Embedding ----------
#     all_blocks = await knwoledge_graph_inst.fetch_all_rule_blocks()
#     inner_ids = {b["id"] for b in all_blocks if b.get("regulation") and "内规" in b["regulation"]}
#     outer_ids = {b["id"] for b in all_blocks if b.get("regulation") and "内规" not in b["regulation"]}
#     desc_map = await knwoledge_graph_inst.fetch_chunk_descriptions(list(inner_ids | outer_ids))
#     embed_only = 0
#
#     async def enrich_by_embed(query_ids: Set[str], target_vdb, target_set: Set[str], reverse=False):
#         nonlocal embed_only
#         for qid in query_ids:
#             q_desc = desc_map.get(qid, {}).get("desc", "")
#             if not q_desc:
#                 continue
#             for rank, hit in enumerate(await target_vdb.query_regulation(q_desc, top_k=embed_top_k)):
#                 hid = hit.get("regulation_id")
#                 certainty = hit.get("certainty", 0.0)
#                 if (not hid) or certainty < embed_sim_cutoff or hid not in target_set:
#                     continue
#                 pair = (qid, hid) if not reverse else (hid, qid)
#                 score_add = embed_weight * certainty * (embed_top_k - rank) / embed_top_k
#                 if pair not in pre_scores:
#                     embed_only += 1
#                 pre_scores[pair] = max(pre_scores.get(pair, 0.0) + score_add, pre_scores.get(pair, 0.0))
#
#     await enrich_by_embed(inner_ids, outer_regulation_vdb, outer_ids, reverse=False)
#     await enrich_by_embed(outer_ids, inner_regulation_vdb, inner_ids, reverse=True)
#     print(f"🔎 向量通道独有命中 {embed_only} 对")
#
#     # ---------- 0C. 融合 ----------
#     refined_pairs = {
#         p for p, s in pre_scores.items() if s >= combined_cutoff
#     }
#     print(f"🚀 融合阈值后 {len(refined_pairs)} 对送入 LLM")
#
#     if not refined_pairs:
#         return 0
#
#
#     # ==== llm_input_pairs2.csv ====
#     llm_rows: List[List[str]] = []
#     for iid, oid in refined_pairs:
#         internal_id, external_id, i_data, o_data = split_internal_external(iid, oid, desc_map)
#         llm_rows.append([
#             extract_clause(i_data.get("desc", "")),
#             extract_clause(o_data.get("desc", ""))
#         ])
#
#     with open("/home/weida/PycharmProjects/laip_graphrag/llm_input_pairs.csv", "w", newline='', encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow(["internal", "external"])
#         writer.writerows(llm_rows)
#     print("[INFO] 已写 llm_input_pairs.csv")
#
#
#     # ---------- 1. 社区报告 ----------
#     comm_ids = {d.get("community_hashid") or d.get("community_id") for d in desc_map.values() if d}
#     comm_ids.discard(None)
#     comm_raw = await community_reports.get_by_ids(list(comm_ids))
#     comm_map = {cid: (r.get("report_string") if isinstance(r, dict) else "") for cid, r in zip(comm_ids, comm_raw)}
#
#     # ---------- 2. LLM 评估 ----------
#     llm_func = global_config["best_model_func"]
#     prompt_tpl = PROMPTS["regulation_consistency_evaluation"]
#     written, batch_num = 0, 0
#     llm_passed: List[Tuple[str, str]] = []
#
#     it = iter(refined_pairs)
#     while (batch := list(islice(it, batch_llm))):
#         llm_tasks = []
#         for iid, oid in batch:
#             internal_id, external_id, i_data, o_data = split_internal_external(iid, oid, desc_map)
#             prompt = prompt_tpl.format(
#                 query=i_data.get("desc", ""),
#                 response=o_data.get("desc", ""),
#                 query_community_report=comm_map.get(i_data.get("community_hashid") or i_data.get("community_id"), ""),
#                 response_community_report=comm_map.get(o_data.get("community_hashid") or o_data.get("community_id"), ""),
#             )
#             llm_tasks.append(llm_func(prompt))
#
#         llm_raw = await asyncio.gather(*llm_tasks)
#         for (iid, oid), rsp in zip(batch, llm_raw):
#             if isinstance(rsp, list):
#                 rsp = rsp[0].get("text", "")
#             try:
#                 data = extract_json_from_response(rsp)
#                 score = float(data["score"])
#                 explanation = data["explanation"]
#
#
#             except Exception:
#                 continue
#             if score < llm_threshold:
#                 continue
#
#             internal_id, external_id, *_ = split_internal_external(iid, oid, desc_map)
#             await knwoledge_graph_inst.upsert_match_with_edge(internal_id, external_id, score, explanation)
#             llm_passed.append((internal_id, external_id))
#             written += 1
#         batch_num += 1
#         print(f"[LLM] Batch {batch_num}: 累计通过 {written}")
#
#
#     print(f"✅ 完成，写入 {written} 条 MATCH_WITH")
#     return written



# async def link_rules(
#     knwoledge_graph_inst,             # BaseGraphStorage
#     community_reports,                # BaseKVStorage
#     inner_regulation_vdb,             # BaseVectorStorage（存内规）
#     outer_regulation_vdb,             # BaseVectorStorage（存外规）
#     global_config: dict,
#     *,
#     batch_llm: int       = 20,        # LLM 并发批量
#     label_weight: float  = 0.7,       # Jaccard 权重 – 标签
#     entity_weight: float = 0.3,       # Jaccard 权重 – 实体
#     jaccard_cutoff: float = 0.25,     # Jaccard 阈值
#     embed_top_k: int     = 5,         # 向量检索 Top-K
#     llm_threshold: float = 7.0,       # LLM 通过分
# ) -> int:
#     """
#     0) Jaccard 预筛
#     1) 双向向量召回再筛
#     2) 社区报告拼装 + LLM 逐对评估
#     3) 写 :内外规匹配 边
#     返回成功写入的边数量
#     """
#
#     # ---------- 0. Jaccard 预筛 ----------
#     pairs: List[Tuple[str, str]] = await knwoledge_graph_inst.fetch_candidate_chunk_pairs(
#         label_weight=label_weight,
#         entity_weight=entity_weight,
#         cutoff=jaccard_cutoff,
#     )
#     if not pairs:
#         print("😅 Jaccard 初筛未命中任何对")
#         return 0
#     print(f"🎯 Jaccard 初筛得到 {len(pairs)} 对，进入向量双向筛…")
#
#     cand_set: Set[Tuple[str, str]] = set(pairs)            # 方便 O(1) 查找
#     inner_ids = {iid for iid, _ in pairs}
#     outer_ids = {oid for _, oid in pairs}
#
#     # ---------- 1.1 取块描述 ----------
#     desc_map = await knwoledge_graph_inst.fetch_chunk_descriptions(
#         list(inner_ids | outer_ids)
#     )  # {chunk_id: {'desc': ..., 'community_hashid': ...}}
#
#     # ---------- 1.2 内→外 向量检索 ----------
#     refined_pairs: Set[Tuple[str, str]] = set()
#     for iid in inner_ids:
#         desc = desc_map.get(iid, {}).get("desc", "")
#
#         print("===============desc===================")
#         print(desc)
#         print("===============desc===================")
#
#         if not desc:
#             continue
#         top_hits = await outer_regulation_vdb.query_regulation(desc, top_k=embed_top_k)
#
#         print("===============top_hits===================")
#         print(top_hits)
#         print("===============top_hits===================")
#
#         for hit in top_hits:
#             oid = hit.get("regulation_id")  # Weaviate 中保存的 chunk id
#             if oid and (iid, oid) in cand_set:
#                 refined_pairs.add((iid, oid))
#
#     # ---------- 1.3 外→内 向量检索 ----------
#     for oid in outer_ids:
#         desc = desc_map.get(oid, {}).get("desc", "")
#         if not desc:
#             continue
#         top_hits = await inner_regulation_vdb.query_regulation(desc, top_k=embed_top_k)
#         for hit in top_hits:
#             iid = hit.get("regulation_id")
#             if iid and (iid, oid) in cand_set:
#                 refined_pairs.add((iid, oid))
#
#     if not refined_pairs:
#         print("😅 向量筛选后已无候选对")
#         return 0
#     print(f"✨ 向量双向筛选保留 {len(refined_pairs)} 对，交给 LLM 决胜…")
#
#
#     # ========== 1.4 嵌入筛选后保存 refined_pairs ==========
#     # 保存 refined_pairs，格式：内规描述前缀、外规描述前缀
#     csv_rows = []
#     for iid, oid in refined_pairs:
#         i_desc = desc_map.get(iid, {}).get("desc", "")
#         o_desc = desc_map.get(oid, {}).get("desc", "")
#         i_clause = extract_clause(i_desc)
#         o_clause = extract_clause(o_desc)
#         csv_rows.append([i_clause, o_clause])
#
#     # 写入 CSV 文件
#     with open("/home/weida/PycharmProjects/laip_graphrag/refined_pairs.csv", "w", encoding="utf-8-sig", newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["inner_clause", "outer_clause"])  # 表头
#         writer.writerows(csv_rows)
#     print(f"已将 {len(csv_rows)} 条嵌入筛选对写入 refined_pairs.csv")
#
#     # ---------- 2. 社区报告 ----------
#     comm_ids = {
#         d.get("community_hashid") or d.get("community_id")
#         for d in desc_map.values() if d
#     }
#     comm_ids.discard(None)
#     comm_raw = await community_reports.get_by_ids(list(comm_ids))
#     comm_map = {
#         cid: (r.get("report_string") if isinstance(r, dict) else "")
#         for cid, r in zip(comm_ids, comm_raw)
#     }
#
#     # ---------- 3. LLM 评估 ----------
#     llm_func   = global_config["best_model_func"]
#     prompt_tpl = PROMPTS["regulation_consistency_evaluation"]
#
#     written = 0
#     it = iter(refined_pairs)
#     while True:
#         batch = list(islice(it, batch_llm))
#         if not batch:
#             break
#
#         llm_tasks = []
#         for iid, oid in batch:
#             i_data = desc_map.get(iid, {})
#             o_data = desc_map.get(oid, {})
#
#             i_comm = i_data.get("community_hashid") or i_data.get("community_id")
#             o_comm = o_data.get("community_hashid") or o_data.get("community_id")
#
#             prompt = prompt_tpl.format(
#                 query                     = i_data.get("desc", ""),
#                 response                  = o_data.get("desc", ""),
#                 query_community_report    = comm_map.get(i_comm, ""),
#                 response_community_report = comm_map.get(o_comm, ""),
#             )
#             llm_tasks.append(llm_func(prompt))
#
#         llm_raw = await asyncio.gather(*llm_tasks)
#
#         for (iid, oid), rsp in zip(batch, llm_raw):
#             if isinstance(rsp, list):           # 兼容缓存返回
#                 rsp = rsp[0].get("text", "")
#             try:
#                 data  = json.loads(rsp)
#                 score = float(data["score"])
#                 expl  = data["explanation"]
#             except Exception:
#                 continue
#
#             if score < llm_threshold:
#                 continue
#
#             await knwoledge_graph_inst.upsert_match_with_edge(iid, oid, score, expl)
#             written += 1
#
#     print(f"✅ 完成：成功写入 MATCH_WITH 边 {written} 条")
#     return written

# 将社区报告按子社区拆分和处理
def _pack_single_community_by_sub_communities(
        community: SingleCommunitySchema,  # 当前社区对象，包含了子社区的ID
        max_token_size: int,  # 最大token大小，用于判断报告是否超出长度
        already_reports: dict[str, CommunitySchema],  # 已处理的社区报告字典
) -> tuple[str, int]:

    # 提取当前社区下的所有子社区（如果子社区已经存在于报告字典中）
    all_sub_communities = [
        already_reports[k] for k in community["sub_communities"] if k in already_reports
    ]

    # 去除重复子社区或无效子社区（如空报告）
    all_sub_communities = [
        c for c in all_sub_communities
        if c.get("report_string") and c.get("report_json")
    ]

    # 按照子社区的“出现频率”（occurrence）排序，优先处理高频出现的子社区
    all_sub_communities = sorted(
        all_sub_communities, key=lambda x: x["occurrence"], reverse=True
    )

    # 使用truncate_list_by_token_size对子社区进行切割，确保每个子社区报告不会超过max_token_size
    may_trun_all_sub_communities = truncate_list_by_token_size(
        all_sub_communities,
        key=lambda x: x["report_string"],  # 使用report_string作为关键字进行切割
        max_token_size=max_token_size,
    )

    # 定义子社区报告需要包含的字段
    sub_fields = ["id", "report", "rating", "importance"]

    # 将子社区的报告字段转为CSV格式
    sub_communities_describe = list_of_list_to_csv(
        [sub_fields]
        + [
            [
                i,  # 子社区索引
                c["report_string"],  # 子社区报告的文本
                c["report_json"].get("rating", -1),  # 子社区报告的评分，默认为-1
                c["occurrence"],  # 子社区出现的频次
            ]
            for i, c in enumerate(may_trun_all_sub_communities)
        ]
    )

    # 准备收集所有相关的节点（entities）和边（relationships）
    already_nodes = []
    already_edges = []
    for c in may_trun_all_sub_communities:
        already_nodes.extend(c["nodes"])  # 将子社区的所有节点添加到already_nodes
        already_edges.extend([tuple(e) for e in c["edges"]])  # 将子社区的所有边（无向）添加到already_edges

    # 返回包含子社区报告、报告字符数、节点和边的集合
    return (
        sub_communities_describe,  # 子社区描述CSV
        len(encode_string_by_tiktoken(sub_communities_describe)),  # 子社区报告的字符数
        set(already_nodes),  # 去重后的节点集合
        set(already_edges),  # 去重后的边集合
    )



# 打包单个社区的描述并处理子社区
async def _pack_single_community_describe(
        knwoledge_graph_inst: BaseGraphStorage,  # 图数据库实例
        community: SingleCommunitySchema,  # 当前社区对象
        max_token_size: int = 12000,  # 最大token大小，避免上下文超限
        already_reports: dict[str, CommunitySchema] = {},  # 已处理的社区报告
        global_config: dict = {},  # 配置参数字典
) -> str:
    # 排序当前社区的节点和边，按顺序处理
    nodes_in_order = sorted(community["nodes"])
    edges_in_order = sorted(community["edges"], key=lambda x: x[0] + x[1])

    # 异步获取所有节点数据
    nodes_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_node(n) for n in nodes_in_order]
    )

    # 异步获取所有边数据
    edges_data = await asyncio.gather(
        *[knwoledge_graph_inst.get_edge(src, tgt) for src, tgt in edges_in_order]
    )

    # 定义节点和边的字段（用于描述）
    node_fields = ["id", "entity", "type", "description", "degree", "file_name"]
    edge_fields = ["id", "source", "target", "description", "rank"]

    # 获取节点数据并附加节点度数
    nodes_list_data = [
        [
            i,  # 节点索引
            node_name,  # 节点名称
            node_data.get("entity_type", "UNKNOWN"),  # 节点类型，默认为"UNKNOWN"
            node_data.get("description", "UNKNOWN"),  # 节点描述，默认为"UNKNOWN"
            await knwoledge_graph_inst.node_degree(node_name),  # 获取节点度数
            node_data.get("file_name", "unknown"),  # 获取文件名
        ]
        for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
    ]

    # 根据度数排序节点列表，度数越高越靠前
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)

    # 如果节点数据超出了最大长度，进行切割
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    # 获取边数据并附加边的度数
    edges_list_data = [
        [
            i,  # 边索引
            edge_name[0],  # 边的起始节点
            edge_name[1],  # 边的终止节点
            edge_data.get("description", "UNKNOWN"),  # 边的描述，默认为"UNKNOWN"
            await knwoledge_graph_inst.edge_degree(*edge_name),  # 获取边的度数
        ]
        for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
    ]

    # 根据度数排序边列表
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)

    # 如果边数据超出了最大长度，进行切割
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    # 检查是否需要使用子社区（当节点和边数超过最大token数时，或者如果force_to_use_sub_communities为True）
    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    # 如果需要使用子社区（或者配置强制使用），则处理子社区
    report_describe = ""
    need_to_use_sub_communities = (
            truncated and len(community["sub_communities"]) and len(already_reports)
    )
    force_to_use_sub_communities = global_config["addon_params"].get(
        "force_to_use_sub_communities", False
    )
    if need_to_use_sub_communities or force_to_use_sub_communities:
        logger.debug(
            f"Community {community['title']} exceeds the limit or you set force_to_use_sub_communities to True, using its sub-communities"
        )
        # 打包子社区的描述信息
        report_describe, report_size, contain_nodes, contain_edges = (
            _pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )
        )

        # 将不包含子社区的节点和边与包含子社区的节点和边合并
        report_exclude_nodes_list_data = [
            n for n in nodes_list_data if n[1] not in contain_nodes
        ]
        report_include_nodes_list_data = [
            n for n in nodes_list_data if n[1] in contain_nodes
        ]
        report_exclude_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) not in contain_edges
        ]
        report_include_edges_list_data = [
            e for e in edges_list_data if (e[1], e[2]) in contain_edges
        ]

        # 如果报告大小超过最大token数，将节点和边分配到不同的部分
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_nodes_list_data + report_include_nodes_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )
        edges_may_truncate_list_data = truncate_list_by_token_size(
            report_exclude_edges_list_data + report_include_edges_list_data,
            key=lambda x: x[3],
            max_token_size=(max_token_size - report_size) // 2,
        )

    # 将节点和边数据转换为CSV格式
    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)

    # 生成最终报告
    return f"""-----Reports-----
```csv
{report_describe}
```
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""

# 将社区报告的JSON格式转换为字符串
def _community_report_json_to_str(parsed_output: dict) -> str:
    """将社区报告的JSON格式解析为Markdown字符串"""

    file_name = parsed_output.get("file_name", "")

    # 从解析的JSON中获取标题，若没有提供标题则默认为"Report"
    title = parsed_output.get("title", "Report")

    # 获取报告的摘要信息
    raw_summary = parsed_output.get("summary", "")
    if isinstance(raw_summary, list):
        summary = "；".join(raw_summary)
    elif isinstance(raw_summary, str):
        summary = raw_summary
    else:
        summary = ""

    summary += f"整个社区的内容来自于文件：{file_name}。"

    # 获取报告中的发现（findings），默认为空列表
    findings = parsed_output.get("findings", [])

    # 提取每个发现的简要描述
    def finding_summary(finding: dict):
        # 如果发现是一个字符串，直接返回
        if isinstance(finding, str):
            return finding
        # 否则返回其中的"summary"字段
        return finding.get("summary")

    # 提取每个发现的详细解释
    def finding_explanation(finding: dict):
        # 如果发现是一个字符串，解释为空
        if isinstance(finding, str):
            return ""
        # 否则返回"explanation"字段
        return finding.get("explanation")

    # 将所有发现汇总成Markdown格式的报告内容
    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in findings
    )

    # 构造并返回最终的社区报告Markdown字符串
    return f"# {title}\n\n{summary}\n\n{report_sections}"

# 生成并保存社区报告
async def generate_community_report(
    community_report_kv: BaseKVStorage[CommunitySchema],      # 存储社区报告的 KV
    knwoledge_graph_inst: BaseGraphStorage,                  # 图数据库实例
    global_config: dict,                                     # 全局配置
    changed_hash_chain: dict[str, set[str]],                 # 本轮变动的哈希
):
    """按 changed_hash_chain 增量生成社区报告。"""
    # ==== 一些准备 ====
    llm_extra_kwargs = global_config["special_community_report_llm_kwargs"]
    use_llm_func = global_config["best_model_func"]
    str2json     = global_config["convert_response_to_json_func"]
    prompt_tpl   = PROMPTS["community_report"]

    # 拿最新 schema（一次就够）
    communities_schema = await knwoledge_graph_inst.community_schema()

    # 供进度条使用
    total_todo = sum(len(s) for s in changed_hash_chain.values())
    processed  = 0

    # ---------- 内部工具：生成单个社区报告 ----------
    async def _gen_one(comm: SingleCommunitySchema,
                       done_reports: dict[str, CommunitySchema]):
        nonlocal processed

        describe = await _pack_single_community_describe(
            knwoledge_graph_inst,
            comm,
            max_token_size=global_config["best_model_max_token_size"],
            already_reports=done_reports,
            global_config=global_config,
        )

        rsp = await use_llm_func(prompt_tpl.format(input_text=describe),
                                 **llm_extra_kwargs)
        data = str2json(rsp)

        processed += 1
        ticker = PROMPTS["process_tickers"][processed % len(PROMPTS["process_tickers"])]
        print(f"{ticker}   {processed}/{total_todo}  "
              f"({processed*100//total_todo}%) communities\r",
              end="", flush=True)
        return data
    # -------------------------------------------------

    # --------- 关键改动：直接按照 diff 来驱动 ---------
    # 先把层级键按数字倒序排列，保证“细 → 粗”处理
    level_keys_desc = sorted(changed_hash_chain,
                             key=lambda x: int(x[1:]),
                             reverse=True)
    logger.info(f"Generating by levels: {level_keys_desc}")

    community_datas: dict[str, CommunitySchema] = {}

    for level_key in level_keys_desc:
        target_hashes = changed_hash_chain.get(level_key, set())
        if not target_hashes:
            logger.info(f"🔕 No changes in {level_key}, skipping...")
            continue

        # 把本层要处理的社区拉出来
        todo_pairs = [
            (h, communities_schema[h])
            for h in target_hashes
            if h in communities_schema            # 理论上都在，保守起见再判一次
        ]

        if not todo_pairs:
            logger.info(f"⚠️ Hashes of {level_key} not present in current schema, skipping…")
            continue

        todo_keys, todo_vals = zip(*todo_pairs)

        # 并发生成
        reports = await asyncio.gather(
            *[_gen_one(c, community_datas) for c in todo_vals]
        )

        # 落库前先归并
        community_datas.update(
            {
                k: {
                    "report_string": _community_report_json_to_str(r),
                    "report_json":   r,
                    **v,
                }
                for k, r, v in zip(todo_keys, reports, todo_vals)
            }
        )

    print()  # 换行，清掉进度条

    # ---------- upsert 到 KV ----------
    await community_report_kv.upsert(community_datas)



# 从多个实体的 cluster 信息中找出最相关的社区报告（可选：只取一个）
async def _find_most_related_community_from_entities(
    node_datas: list[dict],  # 实体节点数据，来自实体向量查询后
    query_param: QueryParam,  # 查询参数，包括层级、Token限制等
    community_reports: BaseKVStorage[CommunitySchema],  # 已保存的社区报告数据
):


    related_communities = []
    for node_d in node_datas:
        # 若实体节点未包含clusters字段，跳过（有些旧数据可能没有）
        if "clusters" not in node_d:
            continue
        # 将实体所属的所有 cluster（社区）解析出来
        related_communities.extend(json.loads(node_d["clusters"]))

    # 过滤只保留不超过当前查询层级的社区，并提取ID
    related_community_dup_keys = [
        str(dp["cluster"])
        for dp in related_communities
        if dp["level"] <= query_param.level
    ]


    # 统计每个社区出现的次数（越多越重要）
    related_community_keys_counts = dict(Counter(related_community_dup_keys))

    # 批量获取社区报告内容
    _related_community_datas = await asyncio.gather(
        *[community_reports.get_by_id(k) for k in related_community_keys_counts.keys()]
    )

    # 过滤掉获取失败（为None）的社区
    related_community_datas = {
        k: v
        for k, v in zip(related_community_keys_counts.keys(), _related_community_datas)
        if v is not None
    }



    # 根据出现次数和社区评分（rating）降序排序
    # related_community_keys = sorted(
    #     related_community_keys_counts.keys(),
    #     key=lambda k: (
    #         related_community_keys_counts[k],
    #         related_community_datas[k]["report_json"].get("rating", -1),
    #     ),
    #     reverse=True,
    # )

    skipped_keys = []

    # 先过滤掉找不到数据的 key，并记录跳过的 key
    valid_keys = []
    for k in related_community_keys_counts:
        if (
                k in related_community_datas and
                isinstance(related_community_datas[k], dict) and
                "report_json" in related_community_datas[k]
        ):
            valid_keys.append(k)
        else:
            skipped_keys.append(k)

    # 打印跳过的 key（可以换成 logger.warning）
    if skipped_keys:
        print(f"⚠️ 跳过以下无效 community key（可能缺失 report_json）: {skipped_keys}")

    # 排序
    related_community_keys = sorted(
        valid_keys,
        key=lambda k: (
            related_community_keys_counts[k],
            related_community_datas[k]["report_json"].get("rating", -1),
        ),
        reverse=True,
    )

    # 最终选中的社区数据（按优先级排序）
    sorted_community_datas = [
        related_community_datas[k] for k in related_community_keys
    ]

    # 控制Token总量，截断社区列表
    use_community_reports = truncate_list_by_token_size(
        sorted_community_datas,
        key=lambda x: x["report_string"],
        max_token_size=query_param.local_max_token_for_community_report,
    )

    # 如果只需要一个（如设置了 `local_community_single_one`），只取第一个
    if query_param.local_community_single_one:
        use_community_reports = use_community_reports[:1]

    return use_community_reports


# 基于实体所属的文本源，以及它们的一跳邻居，找出最相关的文本片段。
async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],  # 实体节点数据
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    # 获取每个实体对应的原始 chunk_id 列表（可能多个）
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]

    # 异步获取所有实体的“一跳邻居边”
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )

    # 提取一跳邻居实体（只取边的target，不管有向无向）
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)

    # 获取一跳邻居实体的节点数据
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # 建立 {entity_name -> 所属chunk集合} 的索引
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }

    # 用于收集最终相关文本块，包含其“关系强度”
    all_text_units_lookup = {}

    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1  # 如果实体的文本chunk和其邻居有重叠chunk，说明重要

            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),  # 获取文本内容
                "order": index,  # 实体在原始输入中顺序
                "relation_counts": relation_counts,  # 相关性得分
            }

    # 警告缺失文本块（chunk_id 不存在）
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")

    # 构造成列表结构（带上chunk_id）
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]

    # 根据“输入顺序优先 + 关系强度倒序”进行排序
    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    # 控制Token总量，截断内容
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.local_max_token_for_text_unit,
    )

    # 最终返回只保留 chunk 本体（去掉元数据）
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]
    return all_text_units


# 基于多个实体，找出它们之间连接的边，并排序选出最重要的关系信息。
async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    # 获取所有实体的一跳边（邻接边）
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )

    all_edges = []  # 所有边的集合
    seen = set()  # 去重集合（无向边）

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))  # 无向边规范化（source, target排序）
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    # 获取边的描述数据
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )

    # 获取边的度数（被连接的次数，用作rank）
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )

    # 组装完整的边数据结构
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None  # 过滤不存在的边
    ]

    # 按照“rank优先 + 权重倒序”进行排序
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    # 控制Token数量，截断描述过长的边
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.local_max_token_for_local_context,
    )

    return all_edges_data


# 从用户的查询中构建上下文：查实体、扩展边、找chunk、拿社区，生成最终 csv 形式的完整上下文
async def _build_local_query_context(
    query,  # 用户输入的问题或指令
    knowledge_graph_inst: BaseGraphStorage,  # 图数据库实例
    entities_vdb: BaseVectorStorage,  # 实体的向量数据库
    community_reports: BaseKVStorage[CommunitySchema],  # 社区报告存储
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 原始文本chunk数据库
    query_param: QueryParam,  # 查询参数，控制截断/返回类型等
):
    # 🔍 向量查询：在实体VDB中查找与query最相关的top_k个实体
    results = await entities_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return None  # 查不到实体直接返回空

    # 🔄 获取实体节点数据
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )

    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")

    # 获取实体节点的连接度（作为排序参考）
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )

    # 拼接成完整的实体信息结构体
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]

    # 🧠 1. 获取相关社区（社区报告）
    use_communities = await _find_most_related_community_from_entities(
        node_datas, query_param, community_reports
    )

    # 📄 2. 获取相关文本块（原文chunk）
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )


    # 🔗 3. 获取实体之间的边（关系）
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )

    # ✅ 打印概览信息
    logger.info(
        f"Using {len(node_datas)} entites, {len(use_communities)} communities, {len(use_relations)} relations, {len(use_text_units)} text units"
    )

    # 📋 构建实体部分的上下文
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    # 📋 构建关系（边）部分的上下文
    relations_section_list = [["id", "source", "target", "description", "weight", "rank"]]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    # 📋 构建社区报告部分
    communities_section_list = [["id", "content"]]
    for i, c in enumerate(use_communities):
        communities_section_list.append([i, c["report_string"]])
    communities_context = list_of_list_to_csv(communities_section_list)

    # 📋 构建原始文本片段部分（包含元信息）
    text_units_section_list = [["id", "content", "page_idx", "file_name", "regulation"]]  # 增加表头

    for i, t in enumerate(use_text_units):
        meta = t.get("meta", {})
        text_units_section_list.append([
            i,
            t["content"],
            # "page_idx: " + str(meta.get("page_idx", "")),  # 页码
            "file_name: " + meta.get("file_name", ""),  # 文件名
            "regulation: " + meta.get("regulation", ""),  # 内规/外规
        ])

    text_units_context = list_of_list_to_csv(text_units_section_list)

    # 拼接所有上下文为完整字符串
    return f"""
-----Reports-----
```csv
{communities_context}
```
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""



# 执行完整的 Local RAG 查询流程 —— 构建上下文 ➜ 输入给LLM ➜ 获取回答
async def local_query(
    query,  # 用户自然语言查询
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    community_reports: BaseKVStorage[CommunitySchema],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:

    use_model_func = global_config["best_model_func"]

    if PREPROCESS_QUERY_WITH_HYDE:
        query = await preprocess_query_with_hyde(query, global_config)

    # 🔧 构建上下文
    context = await _build_local_query_context(
        query,
        knowledge_graph_inst,
        entities_vdb,
        community_reports,
        text_chunks_db,
        query_param,
    )


    print(context)

    if LOCAL_LLM_PRODUCE:
        # 🧪 若只想要上下文，不生成回答（调试用）
        if query_param.only_need_context:
            return context

        # ❌ 没有构建成功，返回失败提示
        if context is None:
            return PROMPTS["fail_response"]

        # 📜 使用提示词模板生成system prompt
        sys_prompt_temp = PROMPTS["local_rag_response"]
        sys_prompt = sys_prompt_temp.format(
            context_data=context,
            response_type=query_param.response_type,  # 允许生成摘要 / 解答 / 推理等
        )

        # 💬 调用LLM生成最终回答
        response = await use_model_func(
            query,
            system_prompt=sys_prompt,
        )
        return response
    else:
        return context


async def _map_global_communities(
    query: str,                          # 用户的查询字符串，用于引导 LLM 分析
    communities_data: list["CommunitySchema"],  # 包含多个社区报告的结构化数据
    query_param: "QueryParam",          # 查询参数对象，包含 max token 限制、模型参数等
    global_config: dict,                # 全局配置，提供模型调用函数和解析函数
) -> dict[str, list[dict]]:             # 返回结构为 {community_id: [点位数据...]}

    """
    把社区报告分批送入大模型处理，返回每个社区的分析点列表。
    保证返回结构中使用真实的 community_id 作为字典的键。
    """

    call_llm = global_config["best_model_func"]  # LLM 推理函数，用于调用模型生成结果
    safe_json = global_config["convert_response_to_json_func"]  # JSON 安全解析函数

    # ---------- 1. 按 token 分批 ----------
    batches, remain = [], communities_data[:]     # 初始化分批列表 batches，remain 为未处理数据
    while remain:
        b = truncate_list_by_token_size(          # 使用 token 限制函数，截取一个子 batch
            remain,
            key=lambda x: x["report_string"],     # 计算 token 时只考虑报告内容
            max_token_size=query_param.global_max_token_for_community_report,
        )
        batches.append(b)                         # 添加当前 batch 到结果中
        remain = remain[len(b):]                  # 从剩余列表中删除已处理部分

    # 定义异步处理单个 batch 的函数
    async def _run(batch: list["CommunitySchema"]) -> dict[str, list[dict]]:
        # ① 构建 CSV 格式输入，包含 community_id + 评分 + 报告内容
        rows = [["community_id", "rating", "importance", "content"]]  # 表头
        for c in batch:
            rows.append([
                c["id"],                                       # 真实社区 ID
                c["report_json"].get("rating", 0),             # 报告中可能已有评分，默认为 0
                c["occurrence"],                               # 社区重要性或出现频率
                c["report_string"].replace("\n", " "),         # 内容文本，统一去掉换行
            ])

        # 插入到 Prompt 模板中
        sys_prompt = PROMPTS["global_map_rag_points"].format(
            context_data=list_of_list_to_csv(rows)  # 将 CSV 列表转为字符串
        )



        # ② 发送给大模型进行推理
        raw = await call_llm(
            query,                                    # 用户查询
            system_prompt=sys_prompt,                # 系统提示 + 内容上下文
            **query_param.global_special_community_map_llm_kwargs,  # 额外模型调用参数
        )


        # ③ 尝试解析模型输出的 JSON
        try:
            data = safe_json(raw)  # 使用安全解析器尝试解析
        except Exception:
            try:
                # 尝试修复 JSON 字符串中的非法反斜杠，再用标准 json.loads
                data = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw))
            except Exception:
                return {}  # 两次解析失败后返回空结果

        if not isinstance(data, dict):  # 模型返回的不是 dict 格式也视为失败
            return {}

        result: dict[str, list[dict]] = {}

        # --- A: 若返回结构为 {community_id: [点位列表]}，直接提取
        for k, v in data.items():
            if k.isdigit() and isinstance(v, list):
                result.setdefault(k, []).extend(v)

        # --- B: 若返回结构为 {"points": [...]}，则按每个点提取 community_id
        if "points" in data and isinstance(data["points"], list):
            for idx, pt in enumerate(data["points"]):
                cid = str(pt.get("community_id", "")).strip()
                if not cid and idx < len(batch):  # 没写明 ID 就按顺序回填
                    cid = batch[idx]["id"]
                if cid:
                    result.setdefault(cid, []).append(pt)

        return result

    # ---------- 2. 合并所有批次结果 ----------
    merged: dict[str, list[dict]] = {}
    for d in await asyncio.gather(*(_run(b) for b in batches)):  # 并发处理所有批次
        for cid, pts in d.items():
            merged.setdefault(cid, []).extend(pts)  # 将所有点位合并到对应社区 ID




    return merged  # 返回总结果


# ========= 全量替换 async global_query =========
async def global_query_normal(
    query: str,
    knowledge_graph_inst: "BaseGraphStorage",
    community_reports: "BaseKVStorage[CommunitySchema]",
    query_param: "QueryParam",
    global_config: Dict[str, Any],
    redis_storages: Optional[Dict[str, "BaseKVStorage"]] = None,
) -> str:

    if PREPROCESS_QUERY_WITH_HYDE:
        query = await preprocess_query_with_hyde(query, global_config)

    # ---------- 0. 过滤社区 ----------
    schema = (await knowledge_graph_inst.community_schema())
    schema = {k: v for k, v in schema.items() if v["level"] <= query_param.level}
    if not schema:
        return PROMPTS["fail_response"]

    sorted_cs = sorted(
        schema.items(),
        key=lambda x: x[1]["occurrence"],
        reverse=True,
    )[: query_param.global_max_consider_community]
    cids = [cid for cid, _ in sorted_cs]

    # ---------- 0-2. 读社区数据 ----------
    raw_datas = await community_reports.get_by_ids(cids)
    id_to_data = {cid: d | {"id": cid} for cid, d in zip(cids, raw_datas) if d}

    # ---------- 0-3. rating 再过滤 ----------
    communities, skipped = [], []
    for cid in cids:
        d = id_to_data.get(cid)
        if not d:
            skipped.append((cid, "missing data"))
            continue
        rating = d.get("report_json", {}).get("rating", 0)
        if rating >= query_param.global_min_community_rating:
            communities.append(d)
        else:
            skipped.append((cid, f"rating={rating}"))
    communities = communities[:200]
    if not communities:
        return PROMPTS["fail_response"]

    # ---------- 1. 支撑点抽取 ----------
    map_pts = await _map_global_communities(
        query, communities, query_param, global_config
    )  # Dict[cid, List[point]]

    # ---------- 1-2. cid ↔ chunk_id 映射 ----------
    cid_to_chunk_ids = {
        cid: data.get("chunk_ids", []) for cid, data in id_to_data.items()
    }
    chunk_to_cid = {
        real_id: cid
        for cid, ids in cid_to_chunk_ids.items()
        for _id in ids
        for real_id in str(_id).split("<SEP>") if real_id
    }

    # ---------- 1-3. 扁平化 pts ----------
    final_pts: List[Dict[str, Any]] = []
    for cid, pts in map_pts.items():
        for pt in pts:
            pt["chunk_ids"] = cid_to_chunk_ids.get(cid, [])
            if "description" in pt:
                final_pts.append(
                    {
                        "community_id": cid,
                        "answer": pt["description"],
                        "score": pt.get("score", 1),
                        "source": pt.get("source", "未知"),
                        "chunk_ids": pt["chunk_ids"],
                    }
                )
    final_pts = [p for p in final_pts if p["score"] > 0]
    if not final_pts:
        return PROMPTS["fail_response"]
    final_pts.sort(key=lambda x: x["score"], reverse=True)
    final_pts = truncate_list_by_token_size(
        final_pts,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )

    # ---------- 2. 句段筛选准备 ----------
    txt_store = redis_storages.get("text_chunks") if redis_storages else None
    embed_func = global_config.get("embedding_func")

    spts_by_cid = defaultdict(list)
    for p in final_pts:
        spts_by_cid[p["community_id"]].append(p)

    topk = getattr(query_param, "embed_topk_per_support", 5)
    min_sim = getattr(query_param, "embed_min_sim", 0.15)
    max_union = getattr(query_param, "embed_max_candidate_chunks", 250)

    community_blocks, community_ids = [], []
    cid_to_spts = {}
    # sentence → file 映射（经过 normalize）
    sent2fname: Dict[tuple[str, str], str] = {}

    for cid, spts in tqdm_asyncio(
        spts_by_cid.items(), desc="📐嵌入筛句"
    ):
        if cid not in id_to_data and not cid.isdigit():
            continue

        # 2-1. 收集 chunk 原文
        seen, src_ids = set(), []
        for sp in spts:
            chunk_ids = sp["chunk_ids"]
            if isinstance(chunk_ids, (str, int)):
                chunk_ids = [str(chunk_ids)]
            for sid in chunk_ids:
                if sid and sid not in seen:
                    seen.add(sid)
                    src_ids.append(sid)

        flat = {sid for g in src_ids for sid in g.split("<SEP>") if sid}
        raws = await txt_store.get_by_ids(list(flat))

        sentences = []
        for r in raws:
            fname = r.get("file_name", "") if r else ""
            if r and r.get("content"):
                for s in split_string_by_multi_markers(
                    clean_str(r["content"]), ["。", "！", "？", "\n"]
                ):
                    norm_s = normalize_sentence(s)
                    sentences.append(norm_s)
                    sent2fname[(cid, norm_s)] = fname

        if not sentences:
            continue

        # 2-2. 嵌入相似度筛句
        keep_idx = set(range(len(sentences)))
        if embed_func:
            try:
                vec_in = [p["answer"] for p in spts] + sentences
                vecs = await embed_func(vec_in)
                sp_v, ck_v = vecs[: len(spts)], vecs[len(spts) :]
                sp_n = np.linalg.norm(sp_v, axis=1, keepdims=True) + 1e-8
                ck_n = np.linalg.norm(ck_v, axis=1) + 1e-8
                sims = sp_v @ ck_v.T / (sp_n * ck_n)
                for i, row in enumerate(sims):
                    keep_idx.update(
                        j
                        for j in np.argpartition(-row, topk)[:topk]
                        if row[j] >= min_sim
                    )
                if len(keep_idx) > max_union:
                    best = np.max(sims[:, list(keep_idx)], axis=0)
                    pair = sorted(
                        zip(keep_idx, best),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:max_union]
                    keep_idx = {idx for idx, _ in pair}
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[Embed] {cid} fallback: {e}")

        sel_sents = [sentences[i] for i in sorted(keep_idx)]
        chunk_block = "\n\n".join(
            f"{i+1}. {t}" for i, t in enumerate(sel_sents)
        )
        spt_block = "\n".join(
            f"- ID: {i}\n  内容: {p['answer']}（来源：{p['source']}）"
            for i, p in enumerate(spts)
        )
        community_blocks.append(
            f"### Community {cid} (ID={cid})\n--多个分析要点--\n{spt_block}\n"
            f"--候选原文段落--\n{chunk_block}\n"
        )
        community_ids.append(cid)
        cid_to_spts[cid] = spts

    if not community_blocks:
        return PROMPTS["fail_response"]

    # ---------- 3. 批次发送 LLM 匹配 ----------
    TOKEN_LIMIT = (
        query_param.global_max_token_for_LLM_match
        - query_param.global_tokens_for_LLM_promt
    )
    tlen = lambda txt: max(1, len(txt) // 3)

    bufs, bufs_cids, buf, buf_c, buf_len = [], [], [], [], 0
    for cid, blk in zip(community_ids, community_blocks):
        l = tlen(blk)
        if buf_len + l > TOKEN_LIMIT and buf:
            bufs.append("\n".join(buf))
            bufs_cids.append(buf_c)
            buf, buf_c, buf_len = [], [], 0
        buf.append(blk)
        buf_c.append(cid)
        buf_len += l
    if buf:
        bufs.append("\n".join(buf))
        bufs_cids.append(buf_c)

    sem = asyncio.Semaphore(GLOBAL_CONCURRENT_LLM_LIMIT)
    call_llm = global_config["best_model_func"]
    convert_json = global_config["convert_response_to_json_func"]

    async def one(i: int, blk: str):
        async with sem:
            try:
                return (
                    await call_llm(
                        PROMPTS[
                            "batch_match_multi_supportPoints_entityContents"
                        ].format(community_blocks=blk)
                    ),
                    i,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[LLM] batch {i} failed: {e}")
                return None, i

    results = await tqdm_asyncio.gather(
        *[one(i, b) for i, b in enumerate(bufs)],
        desc="🧩 LLM 原文匹配",
    )

    # ---------- 3-2. 解析 ----------
    matched_texts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for raw, p_idx in results:
        if not raw:
            continue
        if raw.startswith("```"):
            raw = (
                raw.strip()
                .lstrip("```json")
                .lstrip("```")
                .rstrip("```")
                .strip()
            )
        try:
            parsed = json.loads(
                re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", raw)
            )
        except Exception:
            try:
                parsed = convert_json(raw)
            except Exception:
                continue
        if not isinstance(parsed, dict):
            continue

        for cid_key, mp in parsed.items():
            real_cid = cid_key
            if real_cid not in cid_to_spts and real_cid.isdigit():
                idx = int(real_cid)
                if 0 <= idx < len(bufs_cids[p_idx]):
                    real_cid = bufs_cids[p_idx][idx]
            if real_cid not in cid_to_spts and real_cid in chunk_to_cid:
                real_cid = chunk_to_cid[real_cid]
            if real_cid not in cid_to_spts:
                continue

            if isinstance(mp, dict):
                mp_iter = mp.items()
            elif isinstance(mp, list):
                mp_iter = [("0", mp)]
            else:
                mp_iter = [("0", [mp])]

            for idx_str, matches in mp_iter:
                if isinstance(matches, str):
                    matches = [matches]
                if not isinstance(matches, list):
                    continue
                try:
                    spt_idx = int(idx_str)
                except ValueError:
                    continue
                if spt_idx >= len(cid_to_spts[real_cid]):
                    continue
                sp = cid_to_spts[real_cid][spt_idx]

                for m in matches:
                    # ---- 修改起点 ----
                    if isinstance(m, str):
                        cont_raw = extract_inner_content(m)
                        rating = 5
                    else:
                        cont_raw = extract_inner_content(
                            m.get("content", "")
                        )
                        rating = m.get("rating", 5)

                    if not cont_raw:
                        continue

                    cont_norm = normalize_sentence(cont_raw)

                    # strict key
                    fname = sent2fname.get((real_cid, cont_norm))
                    # fuzzy fallback
                    if not fname:
                        fname = fuzzy_lookup_fname(
                            real_cid, cont_norm, sent2fname
                        )
                    # final fallback
                    if not fname:
                        fname = sp.get("source") or "未知"

                    # 多文件名拆分，只留首个
                    if "," in fname:
                        fname = fname.split(",")[0].strip()

                    matched_texts[real_cid].append(
                        {
                            "content": cont_raw,
                            "file_name": fname,
                            "rating": rating,
                        }
                    )


    # ---------- 4. 汇总 ----------
    points_ctx = "\n".join(
        f"----Analyst----\nCommunity ID: {p['community_id']}"
        f"\nImportance Score: {p['score']}\n{p['answer']}"
        for p in final_pts
    )

    community_paras = {}
    for cid in community_ids:
        raws = matched_texts.get(cid, [])
        seen, uniq = set(), []
        for p in sorted(raws, key=lambda x: x["rating"], reverse=True):
            c = p["content"]
            if c not in seen:
                seen.add(c); uniq.append({**p, "cid": cid})
        community_paras[cid] = uniq



    # 4-2. 选全局 top
    selected, g_seen = [], set()
    for paras in community_paras.values():
        for p in paras:  # 从社区列表里依次找
            if p["content"] not in g_seen:  # 同一句只收一次
                selected.append(p)
                g_seen.add(p["content"])
                break  # 该社区已入选，跳出



    remaining = [p for paras in community_paras.values()
                 for p in paras[1:] if p["content"] not in g_seen]
    remaining.sort(key=lambda x: x["rating"], reverse=True)

    limit = max(len(community_paras), 5)
    for p in remaining:
        if len(selected) >= limit:
            break
        selected.append(p); g_seen.add(p["content"])

    ori_cts = selected

    print(        "--------------Analyst 支持点-------------\n" + points_ctx +
        "\n\n--------------回答格式要求--------------\n" + query_param.response_type +
        "\n\n--------------实体原文内容--------------\n" +
        json.dumps(ori_cts, ensure_ascii=False, indent=2))

    # ---------- 5. 返回 ----------
    if GLOBAL_LLM_PRODUCE:
        sys_p = PROMPTS["global_reduce_rag_response"]
        return await call_llm(
            query,
            sys_p.format(
                report_data=points_ctx,
                response_type=query_param.response_type,
                entity_reference_data=json.dumps(ori_cts, ensure_ascii=False, indent=2)
            ),
        )


    return (
        "--------------Analyst 支持点-------------\n" + points_ctx +
        "\n\n--------------回答格式要求--------------\n" + query_param.response_type +
        "\n\n--------------实体原文内容--------------\n" +
        json.dumps(ori_cts, ensure_ascii=False, indent=2)
    )



async def global_query_rerank(
    query: str,
    knowledge_graph_inst: "BaseGraphStorage",
    community_reports: "BaseKVStorage[CommunitySchema]",
    query_param: "QueryParam",
    global_config: Dict[str, Any],
    redis_storages: Optional[Dict[str, "BaseKVStorage"]] = None,
) -> str:
    """
    基于 rerank 模型的全局查询：
    1. 按社区重要度挑选候选社区；
    2. 调用 _map_global_communities 抽取支撑点；
    3. 为每个支撑点用 rerank 模型在所属社区的原文句子中检索最相关的若干条；
    4. 汇总并调用 LLM 生成最终回答。
    """
    # ---------- 0. 预处理 ----------
    if PREPROCESS_QUERY_WITH_HYDE:
        query = await preprocess_query_with_hyde(query, global_config)


    # ---------- 0-1. 过滤社区 ----------
    schema = await knowledge_graph_inst.community_schema()
    schema = {k: v for k, v in schema.items() if v["level"] <= query_param.level}
    if not schema:
        return PROMPTS["fail_response"]

    sorted_cs = sorted(
        schema.items(), key=lambda x: x[1]["occurrence"], reverse=True
    )[: query_param.global_max_consider_community]
    cids = [cid for cid, _ in sorted_cs]

    # ---------- 0-2. 读取社区报告 ----------
    raw_datas = await community_reports.get_by_ids(cids)
    id_to_data = {cid: d | {"id": cid} for cid, d in zip(cids, raw_datas) if d}

    # ---------- 0-3. rating 过滤 ----------
    communities, skipped = [], []
    for cid in cids:
        d = id_to_data.get(cid)
        if not d:
            skipped.append((cid, "missing data")); continue
        rating = d.get("report_json", {}).get("rating", 0)
        if rating >= query_param.global_min_community_rating:
            communities.append(d)
        else:
            skipped.append((cid, f"rating={rating}"))
    communities = communities[:200]
    if not communities:
        return PROMPTS["fail_response"]

    # ---------- 1. 抽取支撑点 ----------
    map_pts = await _map_global_communities(query, communities, query_param, global_config)
    # {cid: [point,…]}

    # ---------- 1-2. cid ↔ chunk_id 映射 ----------
    cid_to_chunk_ids = {cid: data.get("chunk_ids", []) for cid, data in id_to_data.items()}
    chunk_to_cid = {
        real_id: cid
        for cid, ids in cid_to_chunk_ids.items()
        for _id in ids
        for real_id in str(_id).split("<SEP>") if real_id
    }

    # ---------- 1-3. 扁平化 pts ----------
    final_pts: list[dict] = []
    for cid, pts in map_pts.items():
        for pt in pts:
            pt["chunk_ids"] = cid_to_chunk_ids.get(cid, [])
            if "description" in pt:
                final_pts.append(
                    dict(
                        community_id=cid,
                        answer=pt["description"],
                        score=pt.get("score", 1),
                        source=pt.get("source", "未知"),
                        chunk_ids=pt["chunk_ids"],
                    )
                )
    final_pts = [p for p in final_pts if p["score"] > 0]
    if not final_pts:
        return PROMPTS["fail_response"]

    final_pts.sort(key=lambda x: x["score"], reverse=True)
    final_pts = truncate_list_by_token_size(
        final_pts,
        key=lambda x: x["answer"],
        max_token_size=query_param.global_max_token_for_community_report,
    )

    # ---------- 2. 句子收集 ----------
    txt_store = redis_storages.get("text_chunks") if redis_storages else None
    spts_by_cid = defaultdict(list)
    for p in final_pts:
        spts_by_cid[p["community_id"]].append(p)

    sent2fname: dict[tuple[str, str], str] = {}  # (cid, norm_sentence) -> fname
    cid_sentences: dict[str, list[str]] = {}     # cid -> 所有句子

    for cid, spts in spts_by_cid.items():
        # 收集 chunk 原文
        seen_ids, src_ids = set(), []
        for sp in spts:
            chunk_ids = sp["chunk_ids"]
            chunk_ids = [str(chunk_ids)] if isinstance(chunk_ids, (str, int)) else chunk_ids
            for sid in chunk_ids:
                if sid and sid not in seen_ids:
                    seen_ids.add(sid); src_ids.append(sid)

        flat_ids = {sid for g in src_ids for sid in g.split("<SEP>") if sid}
        raws = await txt_store.get_by_ids(list(flat_ids))

        sentences = []
        for r in raws:
            fname = r.get("file_name", "") if r else ""
            if r and r.get("content"):
                for s in split_string_by_multi_markers(
                    clean_str(r["content"]), ["。", "！", "？", "\n"]
                ):
                    norm_s = normalize_sentence(s)
                    sentences.append(norm_s)
                    sent2fname[(cid, norm_s)] = fname
        cid_sentences[cid] = sentences

    if not cid_sentences:
        return PROMPTS["fail_response"]

    # ---------- 3. rerank 匹配 ----------
    rerank_func = global_config.get("rerank_func")
    top_k = getattr(query_param, "rerank_top_k", 10)
    min_score = getattr(query_param, "rerank_min_score", 0.3)

    matched_texts: dict[str, list[dict]] = defaultdict(list)

    async def _match_one_support(cid: str, sp: dict, idx: int):
        cands = cid_sentences.get(cid, [])
        ranked = await rerank_func(sp["answer"], cands, top_k=top_k)
        # 选 max(3, 高分条数) 个
        high_cnt = sum(1 for _, sc in ranked if sc >= min_score)
        sel_num = max(3, high_cnt)
        for sent, score in ranked[:sel_num]:
            cont_norm = normalize_sentence(sent)
            fname = sent2fname.get((cid, cont_norm)) or sp.get("source") or "未知"
            if "," in fname:
                fname = fname.split(",")[0].strip()
            matched_texts[cid].append(
                {"content": sent, "file_name": fname, "rating": score}
            )

    # 并发执行
    await asyncio.gather(
        *[
            _match_one_support(cid, sp, i)
            for cid, spts in spts_by_cid.items()
            for i, sp in enumerate(spts)
        ]
    )

    if not matched_texts:
        return PROMPTS["fail_response"]

    # ---------- 4. 汇总与去重 ----------
    points_ctx = "\n".join(
        f"----Analyst----\nCommunity ID: {p['community_id']}"
        f"\nImportance Score: {p['score']}\n{p['answer']}"
        for p in final_pts
    )

    community_paras = {}
    for cid, paras in matched_texts.items():
        seen, uniq = set(), []
        for p in sorted(paras, key=lambda x: x["rating"], reverse=True):
            if p["content"] not in seen:
                seen.add(p["content"]); uniq.append({**p, "cid": cid})
        community_paras[cid] = uniq

    # 4-2. 按社区轮询选句，剩余按分排序补足
    selected, g_seen = [], set()
    for paras in community_paras.values():
        for p in paras:
            if p["content"] not in g_seen:
                selected.append(p); g_seen.add(p["content"]); break

    remaining = [
        p for paras in community_paras.values()
        for p in paras[1:] if p["content"] not in g_seen
    ]
    remaining.sort(key=lambda x: x["rating"], reverse=True)

    limit = max(len(community_paras), 5)
    for p in remaining:
        if len(selected) >= limit:
            break
        selected.append(p); g_seen.add(p["content"])

    ori_cts = selected

    # ---------- 5. 生成最终回答 ----------
    if GLOBAL_LLM_PRODUCE:
        sys_p = PROMPTS["global_reduce_rag_response"]
        return await global_config["best_model_func"](
            query,
            sys_p.format(
                report_data=points_ctx,
                response_type=query_param.response_type,
                entity_reference_data=json.dumps(ori_cts, ensure_ascii=False, indent=2),
            ),
        )

    # 若关闭 LLM 生成，返回调试信息
    return (
        "--------------Analyst 支持点-------------\n" + points_ctx +
        "\n\n--------------回答格式要求--------------\n" + query_param.response_type +
        "\n\n--------------实体原文内容--------------\n" +
        json.dumps(ori_cts, ensure_ascii=False, indent=2)
    )




# 执行朴素查询，主要依赖文本块和简单的向量搜索，不涉及复杂的社区分析。
async def naive_query(
    query,  # 用户的查询请求
    chunks_vdb: BaseVectorStorage,  # 文本块的向量数据库
    text_chunks_db: BaseKVStorage[TextChunkSchema],  # 文本块存储
    query_param: QueryParam,  # 查询参数
    global_config: dict,  # 全局配置，包含LLM相关参数
):
    use_model_func = global_config["best_model_func"]

    # 向量查询：在文本块数据库中查找与query最相关的top_k个文本块
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]  # 如果没有找到匹配的文本块，返回失败响应

    # 获取这些文本块的ID，并从数据库中提取内容
    chunks_ids = [r["id"] for r in results]
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    # 根据最大Token限制对文本块进行截断
    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],  # 使用内容长度为截断依据
        max_token_size=query_param.naive_max_token_for_text_unit,  # 最大Token限制
    )

    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")

    # 将截断后的文本块内容拼接成一个大的上下文
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])

    # 如果只需要上下文，返回上下文字符串
    if query_param.only_need_context:
        return section

    # 生成系统提示词，并将其交给LLM生成最终的响应
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response



# =================================HyDe 模式==========================================
async def preprocess_query_with_hyde(query: str,  global_config: dict) -> str:

    use_model_func = global_config["best_model_func"]

    # 📜 使用提示词模板生成system prompt
    sys_prompt_temp = PROMPTS["preprocess_query_with_hyde"]
    sys_prompt = sys_prompt_temp.format(
        user_query=query,
    )

    # 💬 调用LLM生成最终回答
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    return response




# --------------------------------------------------------------------------- #
# ---------------------------- ✨ 辅助工具函数 ✨ ----------------------------- #
# --------------------------------------------------------------------------- #

_TUPLE = PROMPTS["DEFAULT_TUPLE_DELIMITER"]

# 将不同格式的名称（如含引号、空格等）转换为统一的标准格式（即去除引号、空格并转为大写）。
def canonical_name(name: str) -> str:
    """把各种奇怪的名字规整成唯一 Key."""
    # 去除前后的空格和引号，并将所有字母转换为大写
    return clean_str(name).strip('"').strip("'").upper()


# 用于对比字符串，找到最匹配的词汇。这个函数主要用于将输入的词汇与一个允许的词汇表进行比较，并返回最接近的匹配项。
def _best_match(word: str,  # 输入词汇
                vocab: List[str],  # 允许的词汇表
                cutoff: float = .8,  # 相似度阈值，0-1之间，0为完全匹配，1为完全不匹配
                default: str = None) -> str:  # 如果没有匹配项返回的默认值
    # 如果word已经在vocab中，直接返回word
    if word in vocab:
        return word

    # 使用difflib.get_close_matches进行相似度匹配，找到与word最接近的词汇
    cand = difflib.get_close_matches(word, vocab, n=1, cutoff=cutoff)

    # 如果有找到候选词，则返回最匹配的词，否则返回默认值
    return cand[0] if cand else default


# 标准化实体类型。通过调用 canonical_name 先规整实体名称，再用 _best_match 在允许的实体类型列表中找到最匹配的类型。
def normalize_entity_type(raw: str) -> str | None:
    t = canonical_name(raw)  # 规整实体名称
    # 在预定义的实体类型列表中寻找最接近的匹配
    return _best_match(t, [c.upper() for c in ENTITY_TYPES])



# 标准化关系类型。这个函数会清理输入的关系类型（去掉非字母和数字的字符），然后通过 _best_match 查找最接近的关系类型。
def normalize_relation_type(raw: str) -> str:
    # 清理掉非字母、数字和下划线的字符，并转为大写
    t = re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fa5]", "_", raw).strip("_").upper()
    # 在预定义的关系类型列表中找到最接近的匹配
    # return _best_match(t,
    #                    [c.upper() for c in RELATIONSHIP_TYPES],
    #                    default="有关于")  # 如果没有匹配，默认返回"有关于"s
    # 如果清理后的字符串非空，直接返回
    if t:
        return t    

    # 默认返回"有关于"（当原始字符串为空或无效时）
    return "有关于"


# 清理描述中的 <SEP> 和 <think> 标签内容
def clean_description(description: str) -> str:



    # 去掉 <SEP> 字样
    description = description.replace("<SEP>", "")
    # 去掉 <think> 和 </think> 标签之间的内容
    description = re.sub(r"<think>.*?</think>", "", description, flags=re.DOTALL)


    return description

# 判断实体名称是否合法
def is_valid_entity_name(name: str) -> bool:
    name = name.strip().strip('"').strip("'")  # 去除前后空格及引号
    if not name or len(name) <= 1:             # 名称为空或太短
        return False
    if name.isdigit():                         # 不允许纯数字
        return False
    if not re.search(r"[A-Z\u4e00-\u9fa5]", name):  # 必须含有中文或大写英文字母
        return False
    return True

# 标准化实体名为统一格式
def clean_name(name: str) -> str:
    # 去除前后空格、引号，并转换为大写，调用 clean_str 进一步清洗
    return clean_str(name.strip().strip('"').strip("'").upper())


