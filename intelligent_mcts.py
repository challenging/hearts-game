from card import Deck
from rules import get_rating

from mcts import TreeNode, MCTS


class IntelligentTreeNode(TreeNode):
    def __init__(self, parent, prior_p, self_player_idx, player_idx):
        super(IntelligentTreeNode, self).__init__(parent, prior_p, self_player_idx, player_idx)



class IntelligentMCTS(MCTS):
    def __init__(self, policy_value_fn, self_player_idx, c_puct=5):
        super(IntelligentMCTS, self).__init__(policy_value_fn, self_player_idx, c_puct)


    def _post_playout(self, node, trick_nr, state, selection_func):
        is_shootthemoon, scores = False, [0, 0, 0, 0]
        if not state.is_finished:
            probs, scores = self._policy(trick_nr, state)
            action_probs = zip([(card.suit.value, 1 << (card.rank.value-2)) for card in Deck().cards], probs)

            who_is_shooter, t_scores = None, []
            for player_idx, score in enumerate(scores):
                if score == 0:
                    who_is_shooter = player_idx
                else:
                    t_scores.append(player_idx)

            is_shootthemoon = who_is_shooter is not None and all([score > 100 for score in t_scores])

            node.expand(state.start_pos, action_probs)
        else:
            scores, is_shootthemoon = state.score()

        rating = get_rating(self._self_player_idx, scores, is_shootthemoon)
        node.update_recursive(rating)


    def __str__(self):
        return "IntelligentMCTS"
