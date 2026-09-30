# agent_builder.py
from dataclasses import dataclass, field
from os.path import isfile, join
from os import listdir
from pathlib import Path
from typing import List

from .grammar import GrammarEngine
from .semantic import SemanticEngine
from .config import Configuration
from .context import ContextWindow
from .writer_story import WriterParams
from .writer_agent import WriterAgent, ModelInput
from .writer_environment import WriterEnvironment
from .curriculum import Curriculum, CurriculumSentence
from .tokens import TokenPage
from .training import TrainingBatch, TrainingSample

DATA_FOLDER: str = "../triadic-data/toy-system/toy-system-v10/"
MODEL_FILENAME: str = "_model.bin"
TOKENS_FILENAME: str = "_tokens.txt"
OUTPUT_FILENAME: str = "_output.txt"
OUTPUT_NO_PAGING_FILENAME = str = "_output_mlp.txt"

@dataclass
class TrainingBatchBuilder:
    agent: WriterAgent

    def update_context(self, sentence: CurriculumSentence, context: ContextWindow):
        context.start_sentence()
        context.update_narrative_memory(sentence.tokens)
        context.update_entity_tracking(sentence.tokens)

    def build_sentence(self, index: int, sentence: CurriculumSentence, line_position: List[float], context: ContextWindow):
        configuration = self.agent.configuration
        sentence.batch: TrainingBatch = TrainingBatch()
        
        # 1. create model input
        model_input = ModelInput(context, line_position, True, True)
            
        # 2. train for each token
        for tok in sentence.tokens:
            target = self.agent.token_dictionary.add_and_get(tok.text)
            if not target.is_eol(): # do not learn EOL as transition
                self.agent.glp_network.learn(model_input, target, sentence.batch)
                context.add_token(target)

        # 3. context / narrative for each sentence
        self.update_context(sentence, context)
            
        # log statistics for batch
        sentence.batch.show(index)        

    def build_curriculum(self, curriculum_list: List[Curriculum]) -> Curriculum:
        combined_curriculum = Curriculum()
        total_samples: int = 0
        
        for curriculum in curriculum_list:
            line: int = 0
            paragraph_line: int = 0
            paragraph_divider: int = self.agent.configuration.paragraph_divider
            last_line = len(curriculum.sentences) - 1
            context = self.agent.new_context()
            
            for sentence in curriculum.sentences:
                if sentence.starts_paragraph():
                    paragraph_line = 0
                line_position = [line / last_line, paragraph_line / paragraph_divider]                                
                self.build_sentence(line, sentence, line_position, context)
                total_samples += len(sentence.batch.samples)
                line += 1
                paragraph_line += 1
                combined_curriculum.sentences.append(sentence)
                            
        print(f"Total {total_samples} unique samples in {len(combined_curriculum.sentences)} sentences")
        return combined_curriculum

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

    def curriculum_folder_filename(self, environment: WriterEnvironment, curriculum: str) -> str:
        return self.environment_path(environment) + curriculum

    def model_filename(self, environment: WriterEnvironment) -> str:
        return self.environment_path(environment) + environment.prefix + MODEL_FILENAME

    def output_filename(self, params: WriterParams, environment: WriterEnvironment) -> str:
        if params.enable_paging:
            return self.environment_path(environment) + environment.prefix + OUTPUT_FILENAME
        else:
            return self.environment_path(environment) + environment.prefix + OUTPUT_NO_PAGING_FILENAME

    def load_or_create_agent(self, environment: WriterEnvironment) -> WriterAgent:
        name = environment.configuration.name
        if Path(self.model_filename(environment)).is_file():
            print(f"Loading existing agent {name}!")
            agent = WriterAgent.load(environment, self.model_filename(environment))
        else:
            print(f"Creating new agent {name}.")
            agent = WriterAgent(environment, name)
        return agent
        
    def build_curriculum(self, environment: WriterEnvironment, name: str) -> List[Curriculum]:
        folder_filename = self.curriculum_folder_filename(environment, name)
        files = [join(folder_filename, f) for f in listdir(folder_filename)]
        book_files = [f for f in files if isfile(f) and f.endswith(".txt") and not "_tokens" in f]
        print(f"curriculum files = {len(book_files)}")

        curriculum_list = []
        for curriculum_filename in book_files:            
            preprocessed_filename = curriculum_filename.replace(".txt","_tokens.txt")            
            curriculum = Curriculum()
            if Path(preprocessed_filename).is_file():
                curriculum.read_prepocessed(preprocessed_filename, environment)
                print(f"Read preprocessed curriculum from {preprocessed_filename}.")
            else: 
                curriculum.read_curriculum(curriculum_filename, environment)
                curriculum.write_curriculum(preprocessed_filename)
                print(f"Created curriculum from {curriculum_filename}.")
            curriculum_list.append(curriculum)

        return curriculum_list

    def build_environment(self, configuration: Configuration, prefix: str) -> WriterEnvironment:
        environment = WriterEnvironment(configuration, self.grammar, self.semantic, prefix)
        return environment
    
    def train_agent(self, environment: WriterEnvironment, curriculum: List[Curriculum]):
        print("Training agent from curriculum...")
        
        random_epochs = environment.configuration.random_epochs
        print(f"random epochs = {random_epochs}")

        agent: WriterAgent = self.load_or_create_agent(environment)

        agent.add_tracking(curriculum)

        training_builder: TrainingBatchBuilder = TrainingBatchBuilder(agent)
        combined_curriculum: Curriculum = training_builder.build_curriculum(curriculum)
        
        agent.train_curriculum(combined_curriculum, random_epochs)        
        agent.save(self.model_filename(environment))
