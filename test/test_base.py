from model import (
    AskInfo,
    ScenarioInfo,
    TriedStep,
    FlowRule,
)

# 构造测试场景库
def build_demo_scenario() -> ScenarioInfo:
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
            TriedStep(id="A1", desc="若为平台自营且订单未完成，引导用户在订单详情中修改发票信息",
                      slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A2", desc="若为第三方订单，告知用户联系对应卖家处理发票修改问题",
                      slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A3",
                      desc="若为平台自营且订单已完成或暂无法自助修改，告知用户提供正确的修改信息，由客服登记后协助处理",
                      slot_values=["success", "fail"], is_terminal=True),
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
            FlowRule(target_id="E", target_type="end",
                     condition=[["A1=success"], ["A2=success"], ["A3=success"], ["P1=success"]]),
        ],
    )

