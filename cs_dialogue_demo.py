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
    “发票开具 / 发票修改”场景构造 demo 场景库。
    """
    return ScenarioInfo(
        id="invoice_modify",
        name="发票开具 / 发票修改",
        desc="用户在电商平台下单购物，用户发现发票信息填写错误，希望修改发票信息。",
        initial_questions={
            "I1": "我的发票信息填错了，怎么修改？",
        },
        ask_infos=[
            AskInfo(id="Q1", desc="这笔订单是平台自营还是第三方卖家订单？", slot_values=["self", "third_party"]),
            AskInfo(id="Q2", desc="这笔订单目前是还未完成，还是已经完成？", slot_values=["unfinished", "finished"]),
        ],
        tried_steps=[
            TriedStep(id="A1", desc="若为平台自营且订单未完成，引导用户在订单详情中修改发票信息", slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A2", desc="若为第三方订单，告知用户联系对应卖家处理发票修改问题", slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A3", desc="若为平台自营且订单已完成或暂无法自助修改，告知用户提供正确的修改信息，由客服登记后协助处理", slot_values=["success", "fail"], is_terminal=True),
        ],
        fallbacks={
            "P1": "很抱歉，本次未能完成发票修改。建议您保留订单信息和需修改的发票信息，我们将努力协调处理这个问题，在3个工作日内给您一个妥善的回复！"
        },
        flow_rules=[
            # 询问
            FlowRule(target_id="Q1", target_type="ask", condition=[["I1=yes"]]),
            FlowRule(target_id="Q2", target_type="ask", condition=[["Q1=self"]]),
            # 操作
            FlowRule(target_id="A1", target_type="step", condition=[["Q1=self", "Q2=unfinished"]]),
            FlowRule(target_id="A2", target_type="step", condition=[["Q1=third_party"]]),
            FlowRule(target_id="A3", target_type="step", condition=[["Q1=self", "Q2=finished"]]),
            # 兜底
            FlowRule(target_id="P1", target_type="fallback", condition=[["A1=fail"], ["A2=fail"], ["A3=fail"]]),
            # 结束
            FlowRule(target_id="E", target_type="end", condition=[["A1=success"], ["A2=success"], ["A3=success"], ["P1=success"]]),
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