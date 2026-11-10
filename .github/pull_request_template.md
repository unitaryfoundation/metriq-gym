# Description

Briefly explain what changed and why.

If there is no associated issue, or it does not provide this context, include:

- for a bug fix, include a minimal reproducer and expected versus observed behavior.
- for a feature or new method, state its requirement.

Mention dependency changes when applicable. Keep the description concise and include only details that help review.

# Issue ticket number and link

Fixes # (issue)

# Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

# How Has This Been Tested?

Describe validation that CI does not cover, with enough detail to reproduce it.
For example, if you smoke tested the change locally against a simulator,
or a hardware backend, then report the results.

For changes affecting scientific results, explain any effect on benchmark scores or comparability with previous results.

# AI assistance

State whether AI tools were used, name them (including version if recorded), and
describe their role. Briefly describe the human review performed. See
[AI_POLICY.md](https://github.com/unitaryfoundation/metriq-gym/blob/main/AI_POLICY.md);
write `AI use: none` if no AI tools were used.

For agent-driven changes requiring prior maintainer approval under that policy,
link the approval here.

# Checklist:

- [ ] I have read the AI contribution policy and take responsibility for this contribution
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] I have added appropriate regression or feature tests, or explained why they are not applicable
- [ ] I have explained any failed or skipped checks and relevant limitations
- [ ] Any dependent changes have been merged and published in downstream modules (or not applicable)
