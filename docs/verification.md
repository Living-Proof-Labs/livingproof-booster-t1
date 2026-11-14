# Verification and publication

The docs workflow checks generated HTML links and fragments, revision metadata,
and shared brand inclusion, and retains a review artifact for 14 days. It does
not execute robot trials, grade capabilities, or measure software coverage.

Scenario runs are published by
[livingproof-verification](https://github.com/Living-Proof-Labs/livingproof-verification).
This repository publishes the adapter and the stock and Playground configurations.
It does not select a world or a scenario, and it does not store run evidence.
A successful documentation build is not a passing robot grade. This manual does
not embed scenario results or measured coverage. Native CLI/API generation is
not part of this initial manual; the existing source and guide remain available.
These are explicit remaining recipe-adoption gaps, not claims of full compliance.

## Deployment prerequisites and acceptance

The `Repository manual` workflow builds PR previews and publishes only from
`main`, after a successful build. An authorized maintainer must have configured
GitHub Pages to use GitHub Actions and allowed the `github-pages` deployment
environment. The workflow does not change those settings. Merge authorization
is separate from deployment authorization.

The intended URL is https://docs.livingprooflabs.com/livingproof-booster-t1/.
It is not a verified live destination until deployment succeeds. After deployment,
match `build.json` to the deployed commit, follow README to manual, check desktop,
mobile, keyboard navigation, deep-link reload and search for `playground`, and
open a result. Check Documentation home, Source, Help and License destinations.
Only then replace the README's pending source link with the verified live URL
and coordinate the organization directory entry.
