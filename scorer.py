from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice


class Scorer:
    def __init__(self, pred, gt):
        self.pred = pred
        self.gt = gt
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Spice(), "SPICE"),
            # (Meteor(), "METEOR"),
        ]

    def evaluate(self):
        total_scores = {}
        for scorer, method in self.scorers:
            score, _ = scorer.compute_score(self.gt, self.pred)
            if isinstance(method, list):
                for sc, m in zip(score, method):
                    total_scores[m] = sc * 10
            else:
                total_scores[method] = score * 10

        return total_scores


pred = "In the traffic image, there are several road users and objects that are noteworthy. Firstly, there is a white sedan parked on the right side of the road, facing the same direction as the ego car. This vehicle is stationary and does not pose an immediate threat to the ego car's path, but it should be monitored in case it starts moving. Additionally, there are several cars ahead in the same lane as the ego car, all moving in the same direction. The ego car should maintain a safe following distance from these vehicles and be prepared to adjust speed or change lanes if necessary.\n\nA pedestrian is also present on the sidewalk to the right, walking parallel to the road. Although the pedestrian is not on the road, the ego car should be aware of their potential to enter the roadway unexpectedly and be ready to stop or slow down if necessary.\n\nRegarding traffic signs, there is a no parking sign on the right side of the road, indicating that parking is not allowed in that area. This sign is important for the ego car to understand local traffic regulations and to avoid parking in restricted zones.\n\nThe traffic lights ahead show green for the direction the ego car is facing, indicating that the ego car has the right of way to proceed."
gt = "The traffic scene under observation comprises a variety of road users and elements that are crucial for the ego vehicle to consider for safe navigation. Firstly, there is a large concrete mixer truck positioned in the right lane, slightly ahead of the ego vehicle. This truck has its rear extended a bit into the left lane where the ego vehicle is driving, presenting a potential risk of collision due to the possibility of the truck either merging to the left or its rear swinging more into the left lane. Therefore, it is essential for the ego vehicle to maintain a safe following distance and be ready to adjust its speed or position as needed to avoid any accidents.\n\nAdditionally, a sedan is observed in the left lane, traveling in the same direction as that of the ego vehicle. The sedan is moving at a steady speed, prompting the need for the ego vehicle to also maintain a safe following distance. Alternatively, the ego vehicle may consider changing lanes to overtake the sedan if deemed safe and necessary.\n\nRegarding road infrastructure, road markings are visible on the road surface, indicating a straight path for the left lane and a merging or turning path for the right lane. These markings are critical as they guide lane usage and alert the ego vehicle to the possibility of vehicles from the right lane merging into its lane, indicating the need for additional caution and possible maneuvering.\n\nThe scene is devoid of vulnerable road users, traffic lights, traffic cones, and barriers, indicating no immediate concerns from these categories. However, the environmental setting includes trees and lighting columns along the sidewalk and a hill with some constructions on the right side beyond the mixer truck. Although these do not directly influence driving decisions, they are part of the road's environmental context and contribute to the overall situational awareness required for autonomous driving."

scorer = Scorer({"1": [pred]}, {"1": [gt]})
total_scores = scorer.evaluate()
print(total_scores)
