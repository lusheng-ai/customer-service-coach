from model import (
    AskInfo,
    ScenarioInfo,
    TriedStep,
)

# 构造测试场景库
def build_demo_scenario() -> ScenarioInfo:
    return ScenarioInfo(
        id="netcard_nfc_fail",
        name="网证申领/NFC读卡失败",
        desc="用户在申领网证过程中，手机在读卡/刷证步骤无法完成，流程停滞。",
        initial_questions={
            "I1": "身份证识别时直接一点反应都没有怎么办？",
            "I2": "我卡在读卡那一步了，怎么办？",
            "I3": "我手机不具备NFC怎么办？",
        },
        ask_infos=[
            AskInfo(id="Q1", desc="页面提示内容是什么", slot_values=[]),
            AskInfo(id="Q2", desc="手机是否已经打开NFC功能", slot_values=["yes", "no", "unknown"]),
            AskInfo(id="Q3", desc="手机是否具备NFC功能", slot_values=["yes", "no", "unknown"]),
            AskInfo(id="Q4", desc="请问您是在什么场景下使用呢", slot_values=[]),
            AskInfo(id="Q5", desc="按要求操作后还是失败，失败原因是否和之前一样", slot_values=["yes", "no"]),
            AskInfo(id="Q6", desc="读卡是否成功", slot_values=["yes", "no"]),
        ],
        tried_steps=[
            TriedStep(id="A1", desc="指导用户打开手机NFC功能", slot_values=["success", "fail"], is_terminal=False),
            TriedStep(id="A2", desc="指导用户查看手机是否具备NFC功能", slot_values=["success", "fail"],
                      is_terminal=False),
            TriedStep(id="A3", desc="若手机型号不支持NFC，指导用户更换支持NFC的手机", slot_values=["success", "fail"],
                      is_terminal=True),
            TriedStep(id="A4", desc="指导正确的读卡位置，并建议摘取手机保护壳", slot_values=["success", "fail"],
                      is_terminal=True),
            TriedStep(id="A5", desc="如果无法使用网证，指导其使用身份证进行认证解冻微信",
                      slot_values=["success", "fail"], is_terminal=True),
            TriedStep(id="A6", desc="如果无法使用网证，指导其使用手机号码登录爱山东", slot_values=["success", "fail"],
                      is_terminal=True),
        ],
        fallbacks={
            "P1": "很抱歉没能直接帮到您。后续如有疑问可通过APP“在线客服”进行咨询或拨打4001171166服务热线，人工服务时间8:30-20:30，我们将竭诚为您服务。"
        },

        # 此单元测试不需要流程规则
        flow_rules=[],
    )

