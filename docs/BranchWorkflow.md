# Branch workflow

`main` is the stable branch. Make changes in a short-lived branch created from the
latest `main`, and open a pull request back to `main`. A draft pull request is a
good place to run the smoke check and collect feedback while work is in progress.

Before marking a pull request ready:

1. Run `pre-commit` on the files you changed.
2. Check the affected UI or script locally. For training changes, use a real
   preset and record the model, platform, and result in the pull request.
3. Make sure the `smoke` check passes. It checks syntax and a small set of tests
   without downloading model weights or requiring a GPU. It does not replace a
   real training run for changes to the training pipeline.

Merge a ready pull request with **Squash and merge**, then delete its branch.
Start the next branch from the updated `main`. Avoid long-lived integration
branches: the pull request itself is the place to test a change before it
reaches `main`. If several changes must be tested together, use one temporary
integration branch and delete it after the combined pull request is merged.

When bringing in changes from the original OneTrainer repository, fetch
`upstream/master` and merge it into a temporary branch created from `main`.
Open a pull request from that branch and use **Create a merge commit** after
testing. This keeps the upstream commit history connected, so future updates
are easier to merge. Delete the temporary branch after the pull request merges.

Repository settings for maintainers:

- Protect `main` with a rule that requires a pull request and the `smoke` check.
  Do not allow force pushes or deletion of `main`.
- Enable automatic deletion of head branches after merge.
- Do not require an approving review until another maintainer is available;
  GitHub checks and the manual test plan still apply to a solo maintainer.
