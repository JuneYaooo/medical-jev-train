# 调用示例

## 输入与输出案例

下面六个案例使用构造材料，输出由已发布的 medical-jev-4b 在 RTX 5090 上实际运行得到。概率展示到小数点后四位；案例用于说明调用方式，不作为医学能力评测。`answer` 是输入中的选项键，`probabilities` 给出各选项概率。

<details>
<summary>1. 医学教学与题库评测</summary>

输入：

```json
{
  "id": "teaching_diagnosis",
  "request": {
    "state": {
      "clinical_question": "教学病例：患者发热、咳嗽，检测报告为甲型流感病毒核酸阳性，新冠病毒核酸阴性。"
    },
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "本题按检测结果分类：甲型流感核酸阳性选择甲型流感，新冠核酸阳性选择新冠感染，两项均阴性选择证据不足。",
        "criteria": {
          "influenza_a": "甲型流感",
          "covid19": "新冠感染",
          "insufficient_evidence": "证据不足"
        }
      }
    }
  }
}
```

实际输出：

```json
{
  "id": "teaching_diagnosis",
  "status": "ok",
  "answer": "influenza_a",
  "probabilities": {
    "influenza_a": 0.9658,
    "covid19": 0.0173,
    "insufficient_evidence": 0.0169
  }
}
```

</details>

<details>
<summary>2. 护理培训与病例复盘</summary>

输入：

```json
{
  "id": "nursing_priority",
  "request": {
    "state": {
      "nursing_record": "模拟护理记录：患者骶尾部有持续受压记录，当前护理计划要求立即协助变换体位并减压；健康宣教安排在下午。"
    },
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "根据记录，选择当前护理计划中应先执行的一项。",
        "criteria": {
          "pressure_relief": "协助变换体位并减压",
          "health_education": "执行下午的健康宣教",
          "routine_documentation": "整理既往护理记录"
        }
      }
    }
  }
}
```

实际输出：

```json
{
  "id": "nursing_priority",
  "status": "ok",
  "answer": "pressure_relief",
  "probabilities": {
    "pressure_relief": 0.9589,
    "health_education": 0.0205,
    "routine_documentation": 0.0205
  }
}
```

</details>

<details>
<summary>3. 用药与治疗方案研究</summary>

输入：

```json
{
  "id": "medication_contraindication",
  "request": {
    "state": {
      "case": "教学病例：患者明确记录对药物甲过敏。",
      "plans": {
        "plan_a": "方案 A 含药物甲",
        "plan_b": "方案 B 含药物乙",
        "plan_c": "方案 C 含药物丙"
      }
    },
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "本题规则：包含已记录过敏药物的方案标记为禁忌。选择应被标记的方案。",
        "criteria": {
          "plan_a": "方案 A",
          "plan_b": "方案 B",
          "plan_c": "方案 C"
        }
      }
    }
  }
}
```

实际输出：

```json
{
  "id": "medication_contraindication",
  "status": "ok",
  "answer": "plan_a",
  "probabilities": {
    "plan_a": 0.9603,
    "plan_b": 0.0212,
    "plan_c": 0.0184
  }
}
```

</details>

<details>
<summary>4. 随访、宣教与康复演练</summary>

输入：

```json
{
  "id": "followup_plan",
  "request": {
    "state": {
      "discharge_plan": "模拟出院计划：出院后第 7 天门诊复查。",
      "current_record": "今天是出院后第 7 天，本题未报告新增症状。"
    },
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "仅按出院计划选择今天对应的随访安排。",
        "criteria": {
          "scheduled_review": "按计划进行第 7 天门诊复查",
          "delay_one_month": "将首次复查推迟一个月",
          "cancel_followup": "取消后续复查"
        }
      }
    }
  }
}
```

实际输出：

```json
{
  "id": "followup_plan",
  "status": "ok",
  "answer": "scheduled_review",
  "probabilities": {
    "scheduled_review": 0.964,
    "delay_one_month": 0.0175,
    "cancel_followup": 0.0185
  }
}
```

</details>

<details>
<summary>5. 指南规则与试验转介匹配</summary>

输入：

```json
{
  "id": "trial_referral",
  "request": {
    "state": {
      "trial_rules": "虚构研究初筛条件：年龄 18 至 65 岁，标志物 M 阳性，且无列出的排除项。",
      "case": {
        "age": 52,
        "marker_m": "阳性",
        "listed_exclusion_present": false
      }
    },
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "仅依据给出的初筛条件判断是否匹配；初筛匹配不代表正式入组。",
        "criteria": {
          "screening_match": "符合所列初筛条件",
          "screening_mismatch": "不符合所列初筛条件",
          "need_more_information": "给定信息不足"
        }
      }
    }
  }
}
```

实际输出：

```json
{
  "id": "trial_referral",
  "status": "ok",
  "answer": "screening_match",
  "probabilities": {
    "screening_match": 0.9603,
    "screening_mismatch": 0.0198,
    "need_more_information": 0.0199
  }
}
```

</details>

<details>
<summary>6. 医疗消息分类实验</summary>

输入：

```json
{
  "id": "message_classification",
  "request": {
    "state": {
      "message": "您好，我想确认下周三复诊的预约时间。",
      "reported_symptoms": "未报告症状或病情变化。"
    },
    "questions": {
      "decision": {
        "type": "choice",
        "instructions": "按题设规则分类：仅询问预约时间归为常规事务；报告新增不适归为需要临床人员复核；明确求助急救归为紧急求助。",
        "criteria": {
          "routine": "常规事务",
          "clinical_review": "需要临床人员复核",
          "emergency": "紧急求助"
        }
      }
    }
  }
}
```

实际输出：

```json
{
  "id": "message_classification",
  "status": "ok",
  "answer": "routine",
  "probabilities": {
    "routine": 0.9644,
    "clinical_review": 0.0177,
    "emergency": 0.0179
  }
}
```

</details>

