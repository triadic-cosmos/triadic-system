# agent_builder.py
from dataclasses import dataclass, field
from typing import List
from pathlib import Path

from .grammar import GrammarEngine
from .semantic import SemanticEngine
from .config import Configuration
from .context import ContextWindow
from .writer_agent import WriterAgent, ModelInput
from .writer_environment import WriterEnvironment
from .sentence_encoder import SentenceEncoder
from .curriculum import Curriculum, CurriculumStory, CurriculumSentence
from .tokens import TokenPage
from .training import TrainingBatch, TrainingSample

DATA_FOLDER: str = "../triadic-data/toy-system/toy-system-v7/"
MODEL_FILENAME: str = "_model.bin"
TOKENS_FILENAME: str = "_tokens.txt"
OUTPUT_FILENAME: str = "_output.txt"

@dataclass
class TrainingBatchBuilder:
    agent: WriterAgent

    def update_context(self, sentence: CurriculumSentence, context: ContextWindow):
        encoded = sentence.get_encoded(self.agent.glp_network.sentence_encoder)
        context.add_sentence(encoded)
        context.update_narrative_memory(sentence.tokens)

    def build_story(self, index: int, story: CurriculumStory, context: ContextWindow):
        configuration = self.agent.configuration
        story.batch: TrainingBatch = TrainingBatch()
        
        # check if story is trained with a story context or previous context
        if configuration.story_prompt:
            context = self.agent.new_context()
            for sentence in story.sentences:
                self.update_context(sentence, context)
                
        line: int = 0
        for sentence in story.sentences:
            # 1. create model input
            line_number = [line / configuration.line_divider]
            model_input = ModelInput(context, story.embedding, line_number)
            line += 1
            
            # 2. train for each token
            for tok in sentence.tokens:
                target = self.agent.token_dictionary.add_and_get(tok.text)
                self.agent.glp_network.learn(model_input, target, story.batch)
                context.add_token(target)

            # 3. context / narrative for each sentence
            self.update_context(sentence, context)
            
        # log statistics for batch
        story.batch.show(index)        

    def build_curriculum(self, curriculum: Curriculum):
        context = self.agent.new_context()

        index: int = 1
        for story in curriculum.stories:
            self.build_story(index, story, context)
            index += 1

@dataclass
class AgentBuilder:
    configuration: Configuration
    grammar: GrammarEngine = field(init=False)
    semantic: SemanticEngine = field(init=False)
    
    def __post_init__(self):
        self.grammar = GrammarEngine(self.configuration)
        self.semantic = SemanticEngine(self.configuration)
    
    def environment_path(self, environment: WriterEnvironment) -> str:
        return DATA_FOLDER + environment.configuration.name + "/"
        
    def curriculum_filename(self, environment: WriterEnvironment, curriculum: str) -> str:
        return self.environment_path(environment) + curriculum + ".txt"

    def preprocessed_filename(self, environment: WriterEnvironment, curriculum: str) -> str:
        return self.environment_path(environment) + curriculum + TOKENS_FILENAME

    def model_filename(self, environment: WriterEnvironment) -> str:
        return self.environment_path(environment) + environment.prefix + MODEL_FILENAME

    def output_filename(self, environment: WriterEnvironment) -> str:
        return self.environment_path(environment) + environment.prefix + OUTPUT_FILENAME

    def load_or_create_agent(self, environment: WriterEnvironment) -> WriterAgent:
        name = environment.configuration.name
        if Path(self.model_filename(environment)).is_file():
            print(f"Loading existing agent {name}!")
            agent = WriterAgent.load(environment, self.model_filename(environment))
        else:
            print(f"Creating new agent {name}.")
            agent = WriterAgent(environment, name)
        return agent
        
    def build_curriculum(self, environment: WriterEnvironment, name: str) -> Curriculum:
        preprocessed_filename = self.preprocessed_filename(environment, name)
        curriculum = Curriculum()
        if Path(preprocessed_filename).is_file():
            curriculum.read_prepocessed(preprocessed_filename, environment)
            print(f"Read preprocessed curriculum from {preprocessed_filename}.")
        else: 
            curriculum_filename = self.curriculum_filename(environment, name)
            curriculum.read_curriculum(curriculum_filename, environment)
            curriculum.write_curriculum(preprocessed_filename)
            print(f"Created curriculum from {curriculum_filename}.")
        print(curriculum)
        return curriculum

    def build_environment(self, configuration: Configuration, prefix: str) -> WriterEnvironment:
        environment = WriterEnvironment(configuration, self.grammar, self.semantic, prefix)
        return environment
                      
    def train_agent(self, environment: WriterEnvironment, curriculum: Curriculum):
        print("Training agent from curriculum...")
        
        random_epochs = environment.configuration.random_epochs
        print(f"random epochs = {random_epochs}")

        agent: WriterAgent = self.load_or_create_agent(environment)
        agent.build_index_from_curriculum(curriculum)
        print(f"keywords = {len(agent.keyword_map)}")

        training_builder: TrainingBatchBuilder = TrainingBatchBuilder(agent)
        training_builder.build_curriculum(curriculum)
        
        agent.train_curriculum(curriculum, random_epochs)        
        agent.save(self.model_filename(environment))
