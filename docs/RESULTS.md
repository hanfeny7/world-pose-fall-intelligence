# Results gallery

This page is a visual audit trail: it shows what the system sees when it is right, when it is uncertain, and when it needs more work.

## Cross-scene wins

<table>
<tr>
<td width="25%"><img src="assets/results/scene-01-fall17.jpg" alt="Fall17 result"></td>
<td width="25%"><img src="assets/results/scene-02-le2i.jpg" alt="Le2i result"></td>
<td width="25%"><img src="assets/results/scene-03-occu.jpg" alt="Occlusion result"></td>
<td width="25%"><img src="assets/results/scene-04-upfall.jpg" alt="UpFall result"></td>
</tr>
<tr>
<td align="center"><sub>Fall17 · pose evidence</sub></td>
<td align="center"><sub>Le2i · indoor fall</sub></td>
<td align="center"><sub>Occlusion stress case</sub></td>
<td align="center"><sub>UpFall · recovery context</sub></td>
</tr>
<tr>
<td><img src="assets/results/scene-05-caucafall.jpg" alt="CaucaFall result"></td>
<td><img src="assets/results/scene-06-edf.jpg" alt="EDF result"></td>
<td><img src="assets/results/scene-07-ofsyn.jpg" alt="OFSynth result"></td>
<td><img src="assets/results/scene-08-gmd.jpg" alt="GMDCSA24 result"></td>
</tr>
<tr>
<td align="center"><sub>CaucaFall · camera shift</sub></td>
<td align="center"><sub>EDF · low-light room</sub></td>
<td align="center"><sub>OFSynth · synthetic variation</sub></td>
<td align="center"><sub>GMDCSA24 · cross-subject test</sub></td>
</tr>
</table>

## Reading the overlays

- **Green frame** — accepted true-positive or correctly rejected negative window.
- **Red frame** — false-positive or false-negative window retained for error analysis.
- **Pose points and lines** — geometric evidence available to the temporal path.
- **world / future / audio fields** — which evidence channels contributed to the decision.
- **phase** — a coarse temporal interpretation such as `falling`, `fallen`, or `normal_negative`.

## Why show failures?

The red examples are not decoration. They are the boundary conditions that shape thresholding, quality gating, temporal voting, and human review. Publishing them makes the engineering claim more credible and gives future improvements a visible target.

> These images come from internal review artifacts and public/research evaluation scenes. They are not a substitute for a dataset card, subgroup analysis, or safety certification.
