"""Semantically-equivalent paraphrases of solve.py's ANNOTATION_RULES, for
the rule-paraphrase probe in bench/prompt_sensitivity.py (--probe rules).

Each variant keeps exactly the same information as the original: the same 5
labels in the same order, the same linguistic-signal cue phrases per label,
and the same worked Japanese examples verbatim -- only the connective
prose, headers, and structure around them differ. That's the point: if
accuracy moves across variants, it's because the model is sensitive to how
the rules are *phrased*, not because a variant taught it something the
others didn't.

Keeping cue phrases and examples byte-identical across variants also means
a paraphrase can't accidentally make the task easier or harder by adding or
dropping evidence -- only wording/structure changes.
"""

from solve import ANNOTATION_RULES

RULE_PARAPHRASES = {
    "original": ANNOTATION_RULES,
    "checklist": """Use the annotation guideline below to pick the correct label. For each label, check whether the company response contains its typical phrases before assigning it.

"+2" Strong Commitment -- assign this when the response makes a firm, decisive commitment.
  Typical phrases: 「〜します」「〜を実施します」「〜を達成します」 / 「〜を決定しています」「〜は確定しています」 / 「方針を変更する考えはありません」(when directly answering a decision or policy question)
  Sample responses: 「来期は増配を実施します。」 / 「中長期でROE12%を達成します。」 / 「この施策により利益成長は実現できると確信しています。」

"+1" Weak or Qualified Commitment -- assign this when the response leans positive but adds qualification, caution, or limited specificity; it signals direction rather than a finalized decision.
  Typical phrases: 「〜していきたい」「〜を目指しています」 / 「〜と考えています」「〜を見込んでいます」 / conditional or hypothetical expressions (e.g., 「〜であれば」「〜次第で」)
  Sample responses: 「成長投資を進めていきたいと考えています。」 / 「今後も収益は拡大していくと見ています。」 / 「環境が整えば、検討を進める考えです。」

"0" Neutral or Hedged Intent -- assign this when the response is genuinely ambiguous, or is purely clarification/explanation/background with no commitment or refusal about future action.
  Typical phrases: 「〜断定できません」「明確な見通しは示せない」 / 「検討中」「状況を見極める必要がある」 / purely descriptive or explanatory statements providing facts or background
  Sample responses: 「現時点では明確な見通しは示せません。」 / 「様々な見方があり、コメントは差し控えます。」 / 「過去にはこのような取り組みを行ってきました。」(background explanation only)

"-1" Weak Refusal -- assign this when the response declines but the refusal is qualified, conditional, or time-bound, leaving room to revisit later.
  Typical phrases: 「現時点では〜しない」「直ちには考えていない」 / 「今後検討の余地はあるが」 / refusals framed as temporary, conditional, or dependent on future circumstances
  Sample responses: 「現時点では配当方針を変更する考えはありません。」 / 「今中計期間中に見直すことは想定していません。」 / 「足元では難しいと考えていますが、今後は検討します。」

"-2" Strong Refusal -- assign this when the response is a clear, definitive rejection with no visible room for reconsideration.
  Typical phrases: 「〜する予定はありません」 / 「〜は行いません」「〜を否定します」
  Sample responses: 「株式分割を行う予定はありません。」 / 「当該事業への投資は実施しません。」 / 「その想定は当社の方針ではありません。」

""",
    "narrative": """The following guideline explains, for each of the five labels, what kind of company statement it covers and which Japanese phrasing typically marks it.

"+2" (Strong Commitment) covers a clear, decisive statement of firm commitment -- language such as 「〜します」「〜を実施します」「〜を達成します」「〜を決定しています」「〜は確定しています」, or 「方針を変更する考えはありません」 when it directly answers a decision or policy question. For example: 「来期は増配を実施します。」「中長期でROE12%を達成します。」「この施策により利益成長は実現できると確信しています。」

"+1" (Weak or Qualified Commitment) covers a positive or leaning commitment that is still qualified, cautious, or unspecific -- it shows directional intent rather than a finalized decision. Look for language like 「〜していきたい」「〜を目指しています」「〜と考えています」「〜を見込んでいます」, or conditional/hypothetical phrasing such as 「〜であれば」「〜次第で」. For example: 「成長投資を進めていきたいと考えています。」「今後も収益は拡大していくと見ています。」「環境が整えば、検討を進める考えです。」

"0" (Neutral or Hedged Intent) covers genuine ambiguity, or statements that only clarify, explain, or give background without committing to or refusing any future action. Look for language like 「〜断定できません」「明確な見通しは示せない」「検討中」「状況を見極める必要がある」, or purely descriptive/explanatory statements of fact or background. For example: 「現時点では明確な見通しは示せません。」「様々な見方があり、コメントは差し控えます。」「過去にはこのような取り組みを行ってきました。」(background explanation only)

"-1" (Weak Refusal) covers a negative stance that is qualified, conditional, or time-bound, leaving room for reconsideration later. Look for language like 「現時点では〜しない」「直ちには考えていない」「今後検討の余地はあるが」, or any refusal framed as temporary, conditional, or dependent on future circumstances. For example: 「現時点では配当方針を変更する考えはありません。」「今中計期間中に見直すことは想定していません。」「足元では難しいと考えていますが、今後は検討します。」

"-2" (Strong Refusal) covers a clear, definitive rejection with no visible room for reconsideration. Look for language like 「〜する予定はありません」「〜は行いません」「〜を否定します」. For example: 「株式分割を行う予定はありません。」「当該事業への投資は実施しません。」「その想定は当社の方針ではありません。」

""",
    "qa_framing": """For each label below, ask whether the company response matches its description; the listed phrases and examples show what a match looks like.

"+2" (Strong Commitment) -- Does the response make a clear, decisive commitment?
  Matches phrases like: 「〜します」「〜を実施します」「〜を達成します」 / 「〜を決定しています」「〜は確定しています」 / 「方針を変更する考えはありません」(when directly answering a decision or policy question)
  As in: 「来期は増配を実施します。」 / 「中長期でROE12%を達成します。」 / 「この施策により利益成長は実現できると確信しています。」

"+1" (Weak or Qualified Commitment) -- Does the response lean positive but hedge with qualification, caution, or limited specificity, i.e. show direction without a finalized decision?
  Matches phrases like: 「〜していきたい」「〜を目指しています」 / 「〜と考えています」「〜を見込んでいます」 / conditional or hypothetical expressions (e.g., 「〜であれば」「〜次第で」)
  As in: 「成長投資を進めていきたいと考えています。」 / 「今後も収益は拡大していくと見ています。」 / 「環境が整えば、検討を進める考えです。」

"0" (Neutral or Hedged Intent) -- Is the response genuinely ambiguous, or purely clarification/explanation/background with no commitment or refusal about future action?
  Matches phrases like: 「〜断定できません」「明確な見通しは示せない」 / 「検討中」「状況を見極める必要がある」 / purely descriptive or explanatory statements providing facts or background
  As in: 「現時点では明確な見通しは示せません。」 / 「様々な見方があり、コメントは差し控えます。」 / 「過去にはこのような取り組みを行ってきました。」(background explanation only)

"-1" (Weak Refusal) -- Does the response decline, but with the refusal qualified, conditional, or time-bound, leaving room to revisit later?
  Matches phrases like: 「現時点では〜しない」「直ちには考えていない」 / 「今後検討の余地はあるが」 / refusals framed as temporary, conditional, or dependent on future circumstances
  As in: 「現時点では配当方針を変更する考えはありません。」 / 「今中計期間中に見直すことは想定していません。」 / 「足元では難しいと考えていますが、今後は検討します。」

"-2" (Strong Refusal) -- Is the response a clear, definitive rejection with no visible room for reconsideration?
  Matches phrases like: 「〜する予定はありません」 / 「〜は行いません」「〜を否定します」
  As in: 「株式分割を行う予定はありません。」 / 「当該事業への投資は実施しません。」 / 「その想定は当社の方針ではありません。」

""",
    "terse": """Label reference (linguistic signals and example phrasing for each):

+2 Strong Commitment: firm, decisive commitment. Cues: 「〜します」「〜を実施します」「〜を達成します」「〜を決定しています」「〜は確定しています」「方針を変更する考えはありません」(as a direct answer to a decision/policy question). E.g. 「来期は増配を実施します。」「中長期でROE12%を達成します。」「この施策により利益成長は実現できると確信しています。」

+1 Weak/Qualified Commitment: positive but qualified, cautious, or unspecific; direction, not a final decision. Cues: 「〜していきたい」「〜を目指しています」「〜と考えています」「〜を見込んでいます」, conditionals (「〜であれば」「〜次第で」). E.g. 「成長投資を進めていきたいと考えています。」「今後も収益は拡大していくと見ています。」「環境が整えば、検討を進める考えです。」

0 Neutral/Hedged Intent: genuine ambiguity, or explanation/background with no commitment or refusal. Cues: 「〜断定できません」「明確な見通しは示せない」「検討中」「状況を見極める必要がある」, purely descriptive statements. E.g. 「現時点では明確な見通しは示せません。」「様々な見方があり、コメントは差し控えます。」「過去にはこのような取り組みを行ってきました。」(background only)

-1 Weak Refusal: declines, but qualified, conditional, or time-bound -- room to revisit later. Cues: 「現時点では〜しない」「直ちには考えていない」「今後検討の余地はあるが」, temporary/conditional refusals. E.g. 「現時点では配当方針を変更する考えはありません。」「今中計期間中に見直すことは想定していません。」「足元では難しいと考えていますが、今後は検討します。」

-2 Strong Refusal: clear, definitive rejection, no room for reconsideration. Cues: 「〜する予定はありません」「〜は行いません」「〜を否定します」. E.g. 「株式分割を行う予定はありません。」「当該事業への投資は実施しません。」「その想定は当社の方針ではありません。」

""",
}
