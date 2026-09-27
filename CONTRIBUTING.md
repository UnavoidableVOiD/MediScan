# MediScan Repository Development & Release Policy

Status: initial governance requirements. More development, CI/CD, testing, code review, deployment,
security, branching and release rules may be added as the project evolves.

## 1. Repository access

- Team members are added to the GitHub repository as contributors according to their responsibilities.
- Permissions are assigned based on the level of access each member needs.

## 2. Protected branches

`dev` and `prod` are protected. Direct pushes to either branch are prohibited.

Work on your own feature, bugfix or hotfix branch:

```
feature/user-authentication
bugfix/login-validation
hotfix/payment-failure
```

## 3. Pull requests

Every change going into `dev` or `prod` goes through a pull request. Direct pushes such as
`git push origin dev` or `git push origin prod` are not permitted.

```
developer branch
      |  pull request
      v
     dev
      |  pull request
      v
     prod
```

## 4. Approvals

- Every PR targeting `dev` or `prod` needs approval from at least one other developer.
- The author of a PR cannot approve it.
- Important changes may require more than one approval.
- A PR is not merged until all required approvals are in.

## 5. Bypass restrictions

- Normal contributors cannot bypass PR or branch-protection rules.
- Only authorised members with the bypass/administrative role may bypass them, and only when genuinely
  necessary (for example an emergency production fix).

## 6. Production merge schedule

Merges into `prod` normally happen only on the designated weekly production release day. Outside that
day, merges into `prod` are not permitted, except for:

- critical production bugs
- emergency hotfixes
- security-critical fixes
- other changes explicitly authorised by the responsible repository administrator/maintainer

Normal feature work waits for the next release window.

## 7. Production releases and version tags

Every successful merge into `prod` produces a new Git tag using semantic versioning, `vMAJOR.MINOR.PATCH`
(for example `v1.0.0`, `v1.0.1`, `v1.1.0`, `v2.0.0`). The tag must point at the production commit that
was released.

```
PR -> required approvals -> merge into prod -> new version tag -> GitHub Release
```

## 8. Emergency production changes

- Critical bugs and hotfixes may be released outside the normal release day.
- They still follow the PR and approval requirements, unless an authorised maintainer decides an emergency
  bypass is necessary.
- Emergency releases also get a new version tag.
