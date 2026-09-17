#!/bin/bash
# Detects voice-parsed ralph-loop trigger phrases and injects a skill invocation instruction.

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('prompt',''))" 2>/dev/null)

RALPH_RE='ralph|ralf|raw|rough'
LOOP_RE='loop|loup|lupe|lube|loob|fluke'
PATTERN="($RALPH_RE)[- ]?($LOOP_RE)"

if echo "$PROMPT" | grep -iEq "$PATTERN"; then
  TASK=$(echo "$PROMPT" | python3 -c "
import sys, re
pattern = sys.argv[1]
prompt = sys.stdin.read().strip()
task = re.sub(r'(?i)^.*?' + pattern + r'[,.\s]*', '', prompt, count=1).strip()
print(task)
" "$PATTERN")
  python3 -c "
import json, sys
task = sys.argv[1]
ctx = (
  'RALPH_LOOP_TRIGGER: Voice detected ralph-loop invocation. Skip conversational response — '
  'immediately invoke the ralph-wiggum:ralph-loop Skill with these exact args:\n\n'
  '  skill: ralph-wiggum:ralph-loop\n'
  '  args: Read docs/skills/agent-pipeline/CLAUDE.md and pass its full contents as the prompt header, then append:\n\n'
  '---\n'
  'TASK: ' + task + '\n'
  'SUCCESS: all tasks in docs/pipeline/state.md marked verified=true\n'
  'Output PIPELINE_COMPLETE when done.\n\n'
  'Also pass: --completion-promise \"PIPELINE_COMPLETE\" --max-iterations 30'
)
out = {
  'continue': True,
  'hookSpecificOutput': {
    'hookEventName': 'UserPromptSubmit',
    'additionalContext': ctx
  }
}
print(json.dumps(out))
" "$TASK"
else
  echo '{"continue": true}'
fi
