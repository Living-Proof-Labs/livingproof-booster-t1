# Contribute, help, and licensing

Open a [pull request](https://github.com/Living-Proof-Labs/livingproof-booster-t1/pulls)
with a focused change; ask questions and report non-sensitive defects in
[issues](https://github.com/Living-Proof-Labs/livingproof-booster-t1/issues).
No dedicated security-reporting route is established by this manual. Do not
publish sensitive reports in public issues.

The documentation build adopts the documentation recipe and brand from
[collaboration-guidelines v7.10.8](https://github.com/Living-Proof-Labs/collaboration-guidelines/tree/v7.10.8),
resolved to `e8686269446bc727a57a1b9d41f5903542c4ecdb`. This is a scoped docs
adoption, not an upgrade of the existing release workflow or runtime pins.
Current documentation-home and source-directory links are the recipe's bounded
navigation exception; executable dependencies remain pinned.

Edit repository Markdown, not generated output. The README remains the canonical
product guide; the build renders it without maintaining another prose copy.
Tasks are generated with native Task list and summary output. Run the build
before proposing a change. Independent review and live publication verification
remain prerequisites for completed recipe adoption.

The package and this documentation use Apache-2.0; upstream model/controller
licenses remain recorded in the existing guide and profiles. Shared brand CSS
is Apache-2.0, fetched with its recorded checksum. See the License page.

## Released manual and delivery recovery

Pull requests and a **Repository manual** dispatch with a blank `release_tag`
build development documentation with read-only repository permissions. They do
not publish Pages. A successful source release calls the same builder with its
exact version and tag. Publication requires an existing immutable, published
stable GitHub Release whose tag resolves to the checked-out, verified commit.
The version appears in the manual, build metadata and artifact name. The HTML
that passes the link, brand and identity checks is the HTML uploaded to Pages.

To recover delivery, dispatch **Repository manual** on `main` with the existing
`release_tag` (`vX.Y.Z`). This rebuilds and tests the manual only; it cannot create
or modify a release, tag or package. The tag must identify current main, checked
again inside the existing Pages deployment lock. Old releases are not deployed
over a newer main. A missing, draft, mutable, prerelease or mismatched release
fails closed, including a bare tag left by failed publication.

The existing **Release** writer recovery is unchanged. A writer retry is distinct
from manual delivery recovery: the latter never calls semantic-release. If a
release is already published and a Release rerun returns `published=false`, use
**Repository manual** with its explicit tag to recover delivery.

Each build uploads a separate Pages artifact, retained for 14 days. Deployment
selects the successful build's recorded artifact name, including on deploy-only
reruns. Inside the existing deployment lock it retrieves the exact native artifact
ID and exact attempt-qualified name through the same-run artifact runtime, with
one bounded retry per route. Both readback tar SHA-256 values must match the
uploaded tar. It then checks current main immediately before deployment. This
uses the original token permissions and no repository checkout in the Pages role. A missing, expired or unavailable artifact fails delivery; there is no
fallback to another build. Rerun all manual jobs or start an explicit tagged
manual dispatch to rebuild and test the bytes, without calling the writer.
Historical tags without this builder are not backfilled. Adoption changes only
future manual delivery; robot tests, source archives, package versions and release
assets keep their existing boundaries.
