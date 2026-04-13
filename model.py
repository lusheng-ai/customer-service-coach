from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Literal


# ----------------------------
# 枚举类
# ----------------------------
class Stage(str, Enum):
    """
    对话阶段
    - S0: 用户开场
    - S1: 询问阶段
    - S2: 确认阶段（给步骤后等待结果）
    - S3: 兜底阶段（替代方式/兜底方案）
    - S4: 结束阶段
    """
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"

class AgentAct(str, Enum):
    """
    客服行为事件 agent_act 取值
    """
    ASK_INFO = "ask_info"
    GIVE_STEPS = "give_steps"
    CONFIRM_RESULT = "confirm_result"
    FALLBACK = "fallback"
    CLOSE = "close"
    OTHER = "other"

# ----------------------------
# 对话事件类
# ----------------------------
@dataclass(frozen=True)
class AgentEvent:
    """
    客服行为事件
    """
    agent_act: AgentAct
    asked_slots: List[str] = field(default_factory=list)       # Q1..Qn
    provided_steps: List[str] = field(default_factory=list)    # A1..An
    mentioned_facts: List[str] = field(default_factory=list)   # 用于 visible_facts

@dataclass(frozen=True)
class UserEvent:
    """
    用户行为事件
    """
    asked_slots_updates: Dict[str, str] = field(default_factory=dict)  # Q->value
    provided_steps_updates: Dict[str, str] = field(default_factory=dict)  # A->success|fail|unknown
    mentioned_facts: List[str] = field(default_factory=list)  # 用于 visible_facts

# ----------------------------
# 状态机标记
# ----------------------------

@dataclass
class Flags:
    """
    flags.pending_result: 是否等待用户对步骤的执行反馈 :contentReference[oaicite:14]{index=14}
    """
    pending_result: bool = False

# ----------------------------
# 对话状态上下文类
# ----------------------------
@dataclass
class DialogueStateContext:
    """
    对话状态上下文：
    - stage: 当前阶段
    - slots: 槽位信息
    - slots_history: 历史上客服问过哪些I/Q/A
    - flags.pending_result: 等待用户反馈闭环
    - turn_index: 当前对话轮次计数
    - end_reason: 结束原因（进入 S4 必须给）
    - visible_facts: 已出现过的事实集合，约束模拟用户不编造
    """
    stage: Stage = Stage.S0
    flags: Flags = field(default_factory=Flags)
    turn_index: int = 0
    end_reason: str = ""
    slots: Dict[str, str] = field(default_factory=dict)
    slots_history: List[str] = field(default_factory=list)
    visible_facts: Set[str] = field(default_factory=set)

    # 新增：最近一次被“确认完成/更新”的业务节点
    last_resolved_node_ids: List[str] = field(default_factory=list)

    # 新增：当前客服这一轮允许命中的下一跳节点
    expected_node_ids: List[str] = field(default_factory=list)

# ----------------------------
# 场景库子对象
# ----------------------------
@dataclass(frozen=True)
class AskInfo:
    """
    询问信息项（Q项）

    用于定义客服在当前场景下“可以询问哪些关键信息”，对应设计文档中的 Q1...Qn。

    属性说明：
    - id:
        询问项唯一标识，如 "Q1"、"Q2"
    - desc:
        询问项描述，即该问题在业务上的含义
    - slot_values:
        该询问项允许填写的槽位枚举值列表。
        若该项为自然语言描述，则置为空列表 []。

    设计依据：
    文档规定 scenario.ask_infos 的每一项包含 id / desc / slot_values，
    并且当槽位值是自然语言描述时，用空数组表示。:contentReference[oaicite:2]{index=2}
    """
    id: str
    desc: str
    slot_values: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TriedStep:
    """
    操作步骤项（A项）

    用于定义客服在当前场景下“可以给出的处理步骤”，对应设计文档中的 A1...An。

    属性说明：
    - id:
        操作步骤唯一标识，如 "A1"、"A2"
    - desc:
        操作步骤描述
    - slot_values:
        该步骤允许的反馈结果枚举值列表。
        通常为 ["success", "fail"]；
        若未来出现自然语言类步骤反馈，也可置为空列表 []。
    - is_terminal:
        该步骤是否为“结束步骤”。
        若用户执行该步骤后反馈 success，则通常应进入结束阶段。

    设计补充说明：
    文档强制字段只有 id / desc / slot_values。这里额外补充 `is_terminal`
    是因为文档明确说明“是否为结束步骤”会影响对话是否结束，
    也就是状态机在判断 S4 时需要依赖这个信息。:contentReference[oaicite:3]{index=3}
    """
    id: str
    desc: str
    slot_values: List[str] = field(default_factory=list)
    is_terminal: bool = False


@dataclass(frozen=True)
class FlowRule:
    """
    对话流程规则项

    用于定义当前场景下，某个流程节点（Q/A/P/E）在什么条件下可以进入。
    这是状态机进行流程控制、评分、判断客服是否按标准路径处理问题的关键配置。

    属性说明：
    - target_id:
        规则所指向的目标节点标识，如 "Q2"、"A3"、"P1"、"E"
        其中 "E" 为固定结束节点。
    - target_type:
        目标节点类型。
        取值建议限制为：
        - "ask"      : 询问项
        - "step"     : 操作步骤
        - "fallback" : 兜底策略
        - "end"      : 结束节点
    - condition: List[List[str]]
        - 外层列表表示 OR（或条件）
        - 内层列表表示 AND（与条件）
        例子：
        [["A3=success"],["A4=success"], ["A5=success"]] 是A3、A4、A5表达式满足其中一个触发准入条件
        [["Q2=no","Q3=unknown"]] 是 Q2和Q3表达式都满足才能触发准入条件
        [["Q2=no","Q3=unknown"],["A4=success"]] 是Q2和Q3表达式都满足 或者A4表达式满足 可触发准入条件

    设计依据：
    文档规定 scenario.flow_rules 的对象属性包括
    target_id / target_type / condition
    用于定义 Q/A/P 的顺序及准入条件。
    """
    target_id: str
    target_type: Literal["ask", "step", "fallback", "end"]
    condition: List[List[str]] = field(default_factory=list)


# ----------------------------
# 场景库信息
# ----------------------------
@dataclass(frozen=True)
class ScenarioInfo:
    """
    场景库信息对象

    用于完整描述一个业务陪练场景，是分类器、状态机、评分模块的共同输入。

    属性说明：
    - id:
        场景唯一标识，如 "netcard_nfc_fail"
    - name:
        场景名称，如 "网证申领/NFC读卡失败"
    - desc:
        场景描述，对业务问题进行简要说明
    - initial_questions:
        初始问题列表，采用字典结构：
        {
            "I1": "身份证识别时直接一点反应都没有怎么办？",
            "I2": "我卡在读卡那一步了，怎么办？"
        }
        这里保持简单字典结构，符合你的设计目标。
    - ask_infos:
        询问信息项列表，对应 Q1...Qn
    - tried_steps:
        操作步骤项列表，对应 A1...An
    - fallbacks:
        兜底策略列表，采用字典结构：
        {
            "P1": "很抱歉没能直接帮到您..."
        }
    - flow_rules:
        流程规则列表，用于定义 Q/A/P/E 的标准业务流转关系

    为什么这样设计：
    1. 文档要求场景库必须包含：
       场景问题列表、询问信息列表、操作步骤列表、业务流程规则、兜底策略等。:contentReference[oaicite:5]{index=5}
    2. 状态机初始化时需要传入整个场景库对象，
       后续分类器解析、流程控制、结束判断、评分也都依赖它。:contentReference[oaicite:6]{index=6}
    3. 你的前置代码已经明确希望：
       初始问题列表、兜底列表保留为简单字典结构；
       Q/A/流程规则则提升为对象结构，便于约束和代码消费。

    建议补充约束（后续可在 __post_init__ 中校验）：
    - initial_questions 的 key 应为 I1/I2...
    - ask_infos 的 id 应为 Q1/Q2...
    - tried_steps 的 id 应为 A1/A2...
    - fallbacks 的 key 应为 P1/P2...
    - flow_rules.target_id 应引用合法节点，或固定为 E
    """
    id: str
    name: str
    desc: str
    initial_questions: Dict[str, str] = field(default_factory=dict)
    ask_infos: List[AskInfo] = field(default_factory=list)
    tried_steps: List[TriedStep] = field(default_factory=list)
    fallbacks: Dict[str, str] = field(default_factory=dict)
    flow_rules: List[FlowRule] = field(default_factory=list)


# ---------------------------------------
# 用户回复意图对象，用于用户模拟器生成访客的话术
# ---------------------------------------
@dataclass(frozen=True)
class UserReplyIntent:
    """
    用户回复意图：
    - reply_mode:
        initial_question      初始提问
        answer_ask_info       回答客服询问信息
        feedback_step_result  反馈步骤执行结果
        fallback_ack          对兜底方案做确认/致谢
        close_ack             对结束话术做确认/结束
        generic_reply         通用回复
    - agent_chat_msg:
        原始客服话术
    - reply_slot_scope:
        根据客服话术回复的范围
    - allowed_facts:
        当前允许复述的事实集合
    - parse_failed:
        是否出现“客服话术严重偏离流程，无法正常解析标准回复意图”的情况
    """
    reply_mode: str
    agent_chat_msg: str = ""
    reply_slot_scope: List[str] = field(default_factory=list)
    allowed_facts: Set[str] = field(default_factory=set)
    parse_failed: bool = False


# ---------------------------------------
# 客服话术业务流程评分对象
# ---------------------------------------
@dataclass(frozen=True)
class AgentBusinessScoreRecord:
    """
    单轮客服行为的业务评分记录
    含义说明：
    - turn_index:
        当前评分对应的对话轮次（建议取 state_ctx.turn_index + 1，
        因为通常在“状态机 apply_agent_event 前”先调用评分）
    - agent_act:
        本轮客服行为类型
    - touched_node_ids:
        本轮客服实际触达的流程节点，如 Q2 / A1 / P1 / E
    - is_hit:
        本轮行为是否符合场景库配置的业务流程规则
    - score_delta:
        本轮分数变化。命中规则通常为 0；违规时为负数
    - violated_node_ids:
        本轮触达但不满足准入条件的节点
    - expected_node_ids:
        基于当前 state_ctx，理论上“允许进入”的节点集合
    - reason:
        评分说明，便于给出具体扣分依据
    - detail:
        更细粒度的结构化明细，便于后续做报表/前端展示
    """
    turn_index: int
    agent_act: AgentAct
    touched_node_ids: List[str] = field(default_factory=list)

    is_hit: bool = True
    score_delta: int = 0

    violated_node_ids: List[str] = field(default_factory=list)
    expected_node_ids: List[str] = field(default_factory=list)

    reason: str = ""
    detail: Dict[str, List[str] | str | int | bool] = field(default_factory=dict)

