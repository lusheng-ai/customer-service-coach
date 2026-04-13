from __future__ import annotations

import json
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from business_scorer import BusinessScorer
from dialogue_classifier import DialogueClassifier
from model import (
    AskInfo,
    FlowRule,
    ScenarioInfo,
    TriedStep, AgentBusinessScoreRecord, )
from state_machine import DialogueStateMachine
from user_simulator import UserSimulator, UserReplyPlanner


# ============================================================
# Demo用：场景库构造
# ============================================================

def build_demo_scenario() -> ScenarioInfo:
    """
    “网证申领/NFC读卡失败”场景构造 demo 场景库。
    flow_rules 仍待完善，因此这里只放一个最小可运行示例规则集，
    主要用于演示完整流程。
    """
    return ScenarioInfo(
        id="netcard_nfc_fail",
        name="网证申领/NFC读卡失败",
        desc="用户在申领网证过程中，手机在读卡/刷证步骤无法完成，流程停滞。",
        initial_questions={
            "I1": "身份证识别时直接一点反应都没有怎么办？",
            # "I2": "我卡在读卡那一步了，怎么办？",
            # "I3": "我手机不具备NFC怎么办？",
        },
        ask_infos=[
            AskInfo(id="Q1", desc="页面提示内容是什么", slot_values=[]),
            AskInfo(id="Q2", desc="手机是否已经打开NFC功能", slot_values=["yes", "no"]),
            AskInfo(id="Q3", desc="手机是否具备NFC功能", slot_values=["yes", "no", "unknown"]),
            AskInfo(id="Q4", desc="请问您是在什么场景下使用呢", slot_values=["爱山东APP"]),
            AskInfo(id="Q5", desc="按要求操作后还是失败，失败原因是否和之前一样", slot_values=["yes", "no"]),
            AskInfo(id="Q6", desc="读卡是否成功", slot_values=["yes", "no"]),
        ],
        tried_steps=[
            TriedStep(id="A1", desc="指导用户打开手机NFC功能", slot_values=["success", "fail"], is_terminal=False),
            TriedStep(id="A2", desc="指导用户查看手机是否具备NFC功能", slot_values=["success", "fail"], is_terminal=False),
            TriedStep(id="A3", desc="若手机型号不支持NFC，指导用户更换支持NFC的手机", slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A4", desc="指导正确的读卡位置，并建议摘取手机保护壳", slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A5", desc="如果无法使用网证，指导其使用身份证进行认证解冻微信", slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A6", desc="如果无法使用网证，指导其使用爱山东认证", slot_values=["success", "fail"], is_terminal=True),
        ],
        fallbacks={
            "P1": "很抱歉没能直接帮到您。后续如有疑问可通过APP“在线客服”进行咨询或拨打4001171166服务热线，人工服务时间8:30-20:30，我们将竭诚为您服务。"
        },
        # 流程规则
        flow_rules=[
            # 询问信息流程规则
            FlowRule(target_id="Q2", target_type="ask", condition=[["I1=yes"],["Q3=yes"],["A2=success"]]),
            FlowRule(target_id="Q3", target_type="ask", condition=[["I1=yes"]]),
            FlowRule(target_id="Q4", target_type="ask", condition=[["A4=fail","Q5=yes"],["A3=fail"]]),
            FlowRule(target_id="Q5", target_type="ask", condition=[["A4=fail"],["Q6=no"]]),
            FlowRule(target_id="Q6", target_type="ask", condition=[["A1=success"]]),

            # 操作步骤流程规则
            FlowRule(target_id="A1", target_type="step", condition=[["Q2=no"]]),
            FlowRule(target_id="A2", target_type="step", condition=[["Q3=unknown"]]),
            FlowRule(target_id="A3", target_type="step", condition=[["Q3=no"],["A2=fail"]]),
            FlowRule(target_id="A4", target_type="step", condition=[["Q2=yes"]]),
            FlowRule(target_id="A6", target_type="step", condition=[["Q4=%爱山东%"]]),

            # 兜底流程规则
            FlowRule(target_id="P1", target_type="fallback", condition=[["A6=fail"]]),

            # 结束流程规则
            FlowRule(target_id="E", target_type="end", condition=[["A3=success"],["A4=success"], ["A5=success"], ["A6=success"],["Q6=yes"]]),
        ],
    )

# ============================================================
# 对话流程 Demo Runner
# ============================================================

class DialogueDemoRunner:
    """
    完整客服多轮对话流程 demo
    """

    def __init__(self, scenario: ScenarioInfo):
        self.scenario = scenario
        self.state_machine = DialogueStateMachine(scenario)
        self.dialogue_history: List[Dict[str, Any]] = []

    # -----------------
    # 初始问题
    # -----------------
    def pick_initial_question(self) -> tuple[str, str]:
        question_id = random.choice(list(self.scenario.initial_questions.keys()))
        return question_id, self.scenario.initial_questions[question_id]

    # -----------------
    # 输出状态信息
    # -----------------
    def _print_state(self):
        st = self.state_machine.state_ctx
        snapshot = {
            "stage": st.stage.value,
            "turn_index": st.turn_index,
            "pending_result": st.flags.pending_result,
            "end_reason": st.end_reason,
            "slots": st.slots,
            "slots_history": st.slots_history,
            "visible_facts": sorted(list(st.visible_facts)),
            "last_resolved_node_ids": st.last_resolved_node_ids,
            "expected_node_ids": st.expected_node_ids,
        }
        print("\n[STATE]")
        print(json.dumps(snapshot, ensure_ascii=False, indent=2))

    # -----------------
    # 记录对话
    # -----------------
    def _append_history(
        self,
        role: str,
        text: str,
        event: Optional[Any] = None,
    ):
        self.dialogue_history.append(
            {
                "role": role,
                "text": text,
                "event": asdict(event) if event else None,
            }
        )

    # -----------------
    # 结束处理
    # -----------------
    @staticmethod
    def _handle_finish(business_score_records: List[AgentBusinessScoreRecord]):
        print("\n[对话结束]")
        print(business_score_records)
        # print(json.dumps(business_score_records, ensure_ascii=False, indent=2))


    # -----------------
    # 运行 demo（交互式）
    # -----------------
    def run_interactive(self):
        print("=" * 72)
        print(f"场景：{self.scenario.name}")
        print(f"描述：{self.scenario.desc}")
        print("=" * 72)

        # 从场景库中选取初始问题并展示
        init_qid, init_text = self.pick_initial_question()

        # 初始化状态机对话上下文
        self.state_machine.init_session(init_qid)

        # 初始问题进入对话历史
        self.dialogue_history.append(
            {
                "role": "user",
                "text": init_text,
                "event": {
                    "initial_question_id": init_qid
                },
            }
        )

        print(f"\n访客开场（{init_qid}）：{init_text}")
        self._print_state()

        business_score_records = []

        while True:
            # 输入客服话术
            agent_text = input("\n请输入客服话术（输入 exit 退出）：").strip()
            if not agent_text:
                continue
            if agent_text.lower() == "exit":
                print("已退出 demo。")
                break

            # 分类器根据客服的话术、场景库信息解析客服行为，生成客服行为事件
            agent_event = DialogueClassifier.recognize_agent_event(agent_text, self.scenario)
            print(f"\n客服：{agent_text}")
            print("[AgentEvent]")
            print(json.dumps(asdict(agent_event), ensure_ascii=False, indent=2))

            # 对本轮客服行为是否符合业务流程规则进行的评分，记录评分结果和依据
            business_score_record = BusinessScorer.score(agent_event, self.state_machine.state_ctx, self.scenario)
            business_score_records.append(business_score_record)
            print("\n按规则流程进行分数处理：\n" + json.dumps(asdict(business_score_record), ensure_ascii=False, indent=2))

            # 客服回复进入对话历史
            self._append_history(role="agent", text=agent_text, event=agent_event)

            # 将客服行为事件传入状态机并更新状态机信息，状态机迁移对话阶段
            self.state_machine.apply_agent_event(agent_event)
            self._print_state()

            # 调用状态机判断是否对话处于结束阶段，是则结束对话
            if self.state_machine.check_dialogue_should_over():
                self._handle_finish(business_score_records)
                break

            # 传入客服话术、客服行为事件和场景库信息，生成访客话术
            user_reply_intent = UserReplyPlanner.build_intent(self.scenario, agent_event, agent_text)
            user_text = UserSimulator.generate_user_reply(user_reply_intent)

            # 分类器根据访客的话术、场景库信息和客服行为事件解析访客行为，生成访客行为事件
            user_event = DialogueClassifier.recognize_user_event(user_text, self.scenario, agent_event)

            # 访客对话进入对话历史
            self._append_history(role="user", text=user_text, event=user_event)

            # 将访客行为事件传入状态机并更新状态机信息
            self.state_machine.apply_user_event(user_event)

            print(f"\n访客：{user_text}")
            print("[UserEvent]")
            print(json.dumps(asdict(user_event), ensure_ascii=False, indent=2))
            self._print_state()

            # 调用状态机判断是否对话处于结束阶段，是则结束对话
            if self.state_machine.check_dialogue_should_over():
                self._handle_finish(business_score_records)
                break


# ============================================================
# main
# ============================================================

def main():
    scenario = build_demo_scenario()
    runner = DialogueDemoRunner(scenario)
    # 交互式运行
    runner.run_interactive()


if __name__ == "__main__":
    main()