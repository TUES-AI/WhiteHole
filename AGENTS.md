# Coding Agent instructions for this project

You are a ML research coding agent. 
We are currently in research and experimentation phase of our JEPA adaptation paper.
We are multiple people working on this project simultaneously and all our agents come through here.

## Repo artifacts:

1. `Project.md` - this is a long file that describes the project, the problem statement, the approach, and the results. It is a living document that is updated as the project progresses. It references papers and what we get as results, data, models, appraoch from them. Always read this first and update it if during a discussion with the user a direciton changes, a new research paper comes up, results require new experiments and etc.

2. `Science_log.md` - in this file we list every experiment that was run (append/insert only), logged in the following format:
```
DD-HH-MM | Experiment Name 
Experiment Description reference to code, commit and results folder.
Experiment results (as few rows as possible - small table if ablation, single line if single experiment)
<empty line>
---
<empty line>
[next experiment]
```

3. In the Science_log.md there is only a brief. The full experiment logs are in a folder in the same <DD-HH-MM-name> format as the experiment log row in this folder:
`results/`

## Commits - treat them as independant experiments. 

If commits are requested/approved by the user, make sure you updated the repo after all decisive work.

**Every time before you commit**, make sure:
- [ ] You have updated the `Science_log.md` with the new experiment results.
- [ ] You have updated the `Project.md` if there are any changes in the approach, results, or references to papers.
- [ ] State the 2 changes above in the commit description message. Never put a empty commit message.
- [ ] If there is a next step state it in the commit message as well. If there is a blocker (eg need data from collaborator) state it in the commit message as well so on pull the collaborator's coding agent can resolve it. 

** When you pull remote changes, inform the user for the things above really shortly and suggest how you would resolve the probelm if its within your capabilities. If its a research problem, it requires human discussion and decision. **
