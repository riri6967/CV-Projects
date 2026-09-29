# Project Report: Road Lane Detection and Departure Warning

## 1. Introduction
Lane keeping is one of the most basic driving tasks, and one of the first to be automated in Advanced Driver-Assistance Systems (ADAS). This project builds a camera-only lane departure warning using **low-level** image processing. It then adds **high-level** scene understanding (vehicle detection and lane assignment) with a pretrained CNN.

## 2. Dataset
- **Udacity CarND-LaneLines-P1** test data: 6 still images (960×540) and 3 dash-cam clips (1,153 frames total). Clip 3 (`challenge.mp4`) is 1280×720 and includes a curve, a concrete bridge section, tree shadows and the car bonnet.
- Taken on Californian highways on sunny days, with solid and dashed white and yellow markings.
- No lane ground-truth is provided. Section 5 explains how we evaluated despite this.

## 3. Pre-processing and analysis
- **HLS conversion:** lane paint is either very light (white → high L) or strongly yellow (H 10–40, high S). HLS separates these better than RGB under shadows.
- **Gaussian blur** before Canny suppresses asphalt texture edges.
- **ROI masking** removes sky, trees and the opposite carriageway.
- **Slope filtering** (\|slope\| > 0.4) removes horizontal edges such as the bonnet and road seams.
- Output videos were checked to confirm the ROI trapezoid (yellow outline) contains the lanes in all clips.

## 4. Methodology
### 4.1 Low-level pipeline
Colour mask → gray → Gaussian 5×5 → Canny(50,150) → trapezoid ROI → HoughLinesP → slope-sign split → least-squares line per side (x = m·y + c) → EMA smoothing (α = 0.2) → offset = (w/2 − lane centre)/lane width → warning if \|offset\| > 0.15.
The detector learns the lane width at the bottom of the image while both lines are visible. When the car drifts and one line leaves the region of interest, the missing line is inferred from that width (**single-line fallback**). This raised departure-warning recall from 39.6 % to 97.6 % in our tests.

### 4.2 High-level pipeline
YOLOv8n (COCO) → vehicle classes {car, motorcycle, bus, truck}, conf ≥ 0.35 → bottom-centre of each box compared with the lane lines at that row → ego / left / right lane → forward collision warning if an ego-lane box is taller than 18 % of the frame.

## 5. Experiments and results
| Test | Result |
|---|---|
| Lane detection rate (both lines), 3 videos | 100 % / 100 % / 100 % |
| Mean frame-to-frame jitter | 0.56 – 2.28 px |
| False departure warnings while driving normally | 0 % of 1,153 frames |
| Manual check of 40 sampled frames | 40 / 40 correct |
| Simulated drift test (690 samples) | accuracy 98.6 %, precision 100 %, recall 97.6 %, F1 98.8 % |
| Offset estimation error (MAE) | 0.53 % of lane width |
| Speed, low-level (CPU, 2 cores) | 25.8 FPS (960×540), 13.6 FPS (1280×720) |
| Speed, low + high level | 6.7 – 9.5 FPS |

**Simulated drift test:** each test frame is sheared horizontally about the horizon: x' = x + k(y − y_h), with y_h the vanishing point of the detected lanes. This moves the road at the bottom of the image sideways by a known amount, while the vanishing point stays fixed, which is what a real sideways drift looks like on a flat road. The true offset is therefore known exactly, and the warning can be scored with precision and recall.

## 6. Limitations and challenges
- **Straight-line model:** curves are approximated by a straight line. A 2nd-order polynomial on a bird's-eye (perspective-warped) view would fit curves better.
- **Hand-tuned thresholds** (colour, Canny, Hough, ROI) were tuned for sunny daytime highways. Night, rain, worn paint and snow would need new values.
- **No true ground truth:** the evaluation relies on the synthetic drift test and a manual check. A labelled dataset such as TuSimple or CULane would allow pixel-level accuracy.
- The simulated shear also shifts the scenery above the horizon, which is unrealistic but outside the ROI.
- **High-level FCW** was not exercised by these clips (no close lead vehicle), and the box-height distance proxy ignores vehicle size and camera pitch.
- **Lane changes** (`whiteCarLaneSwitch.jpg`) briefly confuse the slope-sign split while the car crosses a line.

## 7. Conclusion
Low-level vision is sufficient for a fast and reliable lane departure warning on clear highways (98.8 % F1). High-level vision adds semantic information (which pixels are vehicles and in which lane they are) that edges and colours cannot provide. Together they form a simple two-function ADAS.
