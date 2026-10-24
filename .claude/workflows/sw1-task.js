export const meta = {
  name: 'sw1-task',
  description: 'Build one SW1 task from plan.md: sonnet implements, haiku runs the tests, sonnet fixes and reviews',
  whenToUse: 'Writing a module and its no-hardware tests for an SW1 task. See .claude/delegation.md.',
  phases: [
    { title: 'Implement', detail: 'write the code and tests from the brief', model: 'sonnet' },
    { title: 'Test', detail: 'run the test command, report failures briefly', model: 'haiku' },
    { title: 'Fix', detail: 'up to max_fix_rounds rounds', model: 'sonnet' },
    { title: 'Review', detail: 'check against plan sections and section 16', model: 'sonnet' },
  ],
}

// args: {task, brief, testCmd, plan_sections?: string[], max_fix_rounds?: number}
const a = args || {}
if (!a.task || !a.brief || !a.testCmd) throw new Error('sw1-task needs args {task, brief, testCmd}')
const sections = (a.plan_sections || []).join(', ') || 'the ones the brief names'
const maxFix = a.max_fix_rounds ?? 2

const BUILD = {
  type: 'object',
  properties: {
    files: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['files', 'notes'],
}
const TEST = {
  type: 'object',
  properties: {
    passed: { type: 'boolean' },
    summary: { type: 'string' },
    failures: { type: 'string' },
  },
  required: ['passed', 'summary'],
}
const REVIEW = {
  type: 'object',
  properties: {
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          line: { type: 'integer' },
          severity: { type: 'string', enum: ['blocker', 'should-fix', 'nit'] },
          problem: { type: 'string' },
        },
        required: ['file', 'severity', 'problem'],
      },
    },
    verdict: { type: 'string' },
  },
  required: ['issues', 'verdict'],
}

phase('Implement')
const build = await agent(
  `Implement plan.md task ${a.task} in this repo (the current directory).
Read only plan.md sections ${sections} (grep for each heading, then read just that range) and the files the brief names.

Brief:
${a.brief}

Write the code and its pytest tests. No real hardware is attached: every hardware path needs the no-hardware mode the plan describes, and the tests use only that mode. Stay inside the folders the brief names. Don't commit or push.
Before finishing, run: ${a.testCmd}
Return the files you created or changed, and notes of at most 120 words (decisions, anything unfinished).`,
  { model: 'sonnet', schema: BUILD, label: `implement ${a.task}`, phase: 'Implement' },
)

const runTests = (label) =>
  agent(
    `Run this from the repo root and report the result: ${a.testCmd}
Don't edit any files. If it fails, put the failing test names and the key error lines (at most 40 lines) in failures.`,
    { model: 'haiku', effort: 'low', schema: TEST, label, phase: 'Test' },
  )

phase('Test')
let test = await runTests('test')
let round = 0
while (test && !test.passed && round < maxFix) {
  round++
  phase('Fix')
  await agent(
    `The tests for plan.md task ${a.task} fail. Fix the code, or a test that contradicts plan.md sections ${sections}, then rerun: ${a.testCmd}
Failures:
${test.failures || test.summary}
Don't commit. Return at most 80 words on what you changed.`,
    { model: 'sonnet', label: `fix ${round}`, phase: 'Fix' },
  )
  test = await runTests(`retest ${round}`)
}
if (test && !test.passed) log(`Tests still failing after ${round} fix rounds`)

phase('Review')
const review = await agent(
  `Review the uncommitted changes for plan.md task ${a.task}: git diff, plus the untracked files in git status.
Check them against plan.md sections ${sections} and the rules in section 16:
- matches the section 9 interfaces exactly (names, arguments, return values, exit codes);
- the no-hardware mode runs with no hardware attached;
- nothing moves real hardware by default, and speeds respect the robot.yaml caps;
- no hard-coded ports, camera indices or thresholds; no secrets.
Report only real problems. Don't edit files.`,
  { model: 'sonnet', schema: REVIEW, label: `review ${a.task}`, phase: 'Review' },
)

return { task: a.task, files: build?.files, notes: build?.notes, tests: test, fix_rounds: round, review }
