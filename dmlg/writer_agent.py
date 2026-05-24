# writer_agent.py
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from collections import defaultdict
import random
import pickle

from .config import Configuration
from .writer_environment import WriterEnvironment
from .writer_story import WriterStory, WriterSentence
from .tokens import Token, TokenDictionary, TokenLogit
from .context import ContextWindow, ModelInput
from .paged_network import PagedNetwork, TrainingBatch
from .transition_map import TransitionMap
from .curriculum import Curriculum, CurriculumStory, CurriculumSentence
from .grammar import GrammarEngine
from .semantic import SemanticEngine

@dataclass
class WriterAgent:
    environment: WriterEnvironment
    id: str
    
    keyword_map: Dict[str, List[List[float]]] = field(init=False)
    keyword_count: Dict[str, int] = field(init=False)
    rng: random.Random = field(init=False)
    configuration: Configuration = field(init=False)
    transition_map: TransitionMap = field(init=False)
    paged_network: PagedNetwork = field(init=False)
    token_dictionary: TokenDictionary = field(init=False)

    training_count: int = 0

    def __init__(self, environment: WriterEnvironment, id: str):
        self.environment = environment
        self.configuration = self.environment.configuration
        self.id = id

        self.rng = random.Random()

        self.transition_map = TransitionMap()

        # PagedNetwork verwacht transition_map
        self.paged_network = PagedNetwork(
            configuration=self.configuration,
            transition_map=self.transition_map
        )

        self.token_dictionary = TokenDictionary()
        self.keyword_map = dict()
        self.keyword_count = dict()

    def __str__(self):
        return f"[{self.id}] trainings = {self.training_count}, pages = {self.paged_network.token_pages}"
        
    # ------------------------------------------------------------
    # Learning and curriculum training
    # ------------------------------------------------------------
        
    def train_curriculum(self, curriculum: Curriculum, epochs: int, explore: bool):
        batch = TrainingBatch()
        # Reuse context for all epochs
        context_window = ContextWindow(self.configuration)
        
        for epoch in range(1, epochs + 1):
            story = curriculum.get_random_story(self.rng)
            size = len(story.sentences) - 1
            line = 0;
            for sentence in story.sentences:
                model_input: ModelInput = \
                    ModelInput(context_window, story.embedding, line / size)
                for target in sentence.tokens:
                    self.training_count += 1
                    self.paged_network.learn(model_input, target, batch)
                    model_input.window.add_token(target)
                context_window.add_sentence(sentence.encoded)
                line += 1
            if epoch % self.environment.configuration.epochs_step == 0:
                print(epoch)
                self.learn_batch(batch, explore)
                batch = TrainingBatch()
        
        self.learn_batch(batch, explore)
        self.show()

    def learn_batch(self, batch: TrainingBatch, explore: bool):
        if explore:
            return
        if len(batch.samples) == 0:
            return
        self.paged_network.learn_batch(batch)

    def learn_stories(self, teacher):
        stories = []

        story_target = self.configuration.story_training

        for i in range(story_target):
            story = teacher.write_story(False, False, f"LEARN-{i}")
            stories.append(story)

        for story in stories:
            batch = TrainingBatch([])
            for sentence in story.sentences:
                self.learn_sentence(sentence.tokens, sentence.line, batch)
            self.learn_batch(batch)

    # ------------------------------------------------------------
    # Curriculum story indexing
    # ------------------------------------------------------------

    def build_index_from_curriculum(self, curriculum: Curriculum):
        km = defaultdict(set) 
        kc = defaultdict(int)

        for story in curriculum.stories:
            emb = tuple(story.embedding)  
            kws = story.keywords

            for kw in kws:
                km[kw].add(emb)            
                kc[kw] += 1

        self.keyword_map = {k: list(v) for k, v in km.items()}
        self.keyword_count = dict(kc)

    def score_embeddings(self, keywords: Set[str]) -> Dict[tuple, float]:
        """
        Returns: embedding(tuple) -> score
        Score = sum( 1 / keyword_count[keyword] ) for all keywords the embedding belongs to.
        """
        scores = defaultdict(float)

        for kw in keywords:
            if kw not in self.keyword_map:
                continue
            weight = 1.0 / self.keyword_count[kw]
            
            for emb in self.keyword_map[kw]:
                emb_key = tuple(emb)
                scores[emb_key] += weight

        return scores

    def choose_best_embedding(self, keywords: Set[str]) -> List[float]:
        # ------------------------------------------------------------
        # 1. No keywords → random keyword → random embedding
        # ------------------------------------------------------------
        if not keywords:
            if not self.keyword_map:
                return None

            # random keyword
            kw = self.rng.choice(list(self.keyword_map.keys()))

            # random embedding for that keyword
            return list(self.rng.choice(self.keyword_map[kw]))

        # ------------------------------------------------------------
        # 2. Normal scoring using keywords
        # ------------------------------------------------------------
        scores = self.score_embeddings(keywords)
        if not scores:
            return None

        best_score = max(scores.values())

        # all embeddings with highest score
        candidates = [list(emb) for emb, sc in scores.items() if sc == best_score]

        # random choice from top scores
        return self.rng.choice(candidates)

    # ------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------

    def show(self, full = False):
        print(f"training count = {self.training_count}")
        print(f"page count = {len(self.paged_network.page_list)}")
        if full:
            sizes = []
            for page in self.paged_network.page_list:
                sizes.append(page.get_size_text())
            print(sizes)

    # ------------------------------------------------------------
    # Network optimizer
    # ------------------------------------------------------------

    def optimize(self):
        self.paged_network.optimize()
        self.training_count = 0 # reset counter

    # ------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------

    def save(self, path: str):
        state = {
            "id": self.id,
            "config": self.configuration,
            "token_dictionary": self.token_dictionary,
            "transition_map": self.transition_map,
            "paged_network": self.paged_network,
            "training_count": self.training_count,
            "keyword_map": self.keyword_map,
            "keyword_count": self.keyword_count,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    @staticmethod
    def load(environment: WriterEnvironment, path: str) -> "WriterAgent":
        with open(path, "rb") as f:
            state = pickle.load(f)

        agent = WriterAgent(environment, state["id"])
        agent.configuration = state["config"]
        agent.token_dictionary = state["token_dictionary"]
        agent.transition_map = state["transition_map"]
        agent.paged_network = state["paged_network"]
        agent.training_count = state["training_count"]
        agent.keyword_map = state["keyword_map"]
        agent.keyword_count = state["keyword_count"]
        
        print(f"Loaded agent {agent.id}.")
        agent.show()
        return agent
