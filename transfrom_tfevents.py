# coding: utf-8
from tensorboard.backend.event_processing import event_accumulator

import pandas as pd


def readEvent(event_path, scalarName):
    """
    读tensorboard生成的event文件中指定的标量值
        event_path:event文件路径
        scalarName：要操作的标量名称
    """
    event = event_accumulator.EventAccumulator(event_path)
    event.Reload()
    print("\033[1;34m数据标签：\033[0m")
    print(event.Tags())
    print("\033[1;34m标量数据关键词：\033[0m")
    print(event.scalars.Keys())
    value = event.scalars.Items(scalarName)
    print("你要操作的scalar是：", scalarName)
    return value


def exportToexcel(scalarNameList, excelName):
    """
    将不同的标量数据导入到同一个excel中，放置在不同的sheet下
        注：excel中sheet名称的命名不能有：/\\?*这些符号
    """
    writer = pd.ExcelWriter(excelName)
    for i in range(len(scalarNameList)):
        scalarName = scalarNameList[i]
        scalarValue = readEvent(event_path, scalarName)
        data = pd.DataFrame(scalarValue)
        if scalarName == "rollout/agent_0/SAC_1/reward":
            scalarName = "reward"
        if scalarName == "training/agent_0/SAC_1/policy_loss":
            scalarName = "policy_loss"
        data.to_excel(writer, sheet_name=scalarName)
    writer.save()
    print("数据保存成功")


if __name__ == "__main__":
    event_path = "logs/Gym/share/sac_hopper_1627785355.208333/events.out.tfevents.1627791993.gpu30"
    scalarNameList = ["rollout/agent_0/SAC_1/reward", "training/agent_0/SAC_1/policy_loss"]

    excelName = "data.xlsx"
    exportToexcel(scalarNameList, excelName)
