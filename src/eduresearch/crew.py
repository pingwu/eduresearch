from typing import List
from pydantic import BaseModel, Field
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from crewai_tools import SerperDevTool, WebsiteSearchTool, YoutubeVideoSearchTool, YoutubeChannelSearchTool

class ProgramDetails(BaseModel):
    name: str = Field(..., description="Name of the program or provider")
    description: str = Field(..., description="Short description of the program")
    curriculum_structure: str = Field(None, description="Details about the curriculum structure")
    funding_mechanisms: str = Field(None, description="Details about available funding opportunities")
    outcomes: str = Field(None, description="Reported job placement rates or other success metrics")
    pricing: str = Field(None, description="Cost structure of the program")
    compliance: str = Field(None, description="Compliance requirements for the program")
    completion_rates: float = Field(None, description="Reported program completion rates (if applicable) or studen't feedback rating")
    accreditation: str = Field(None, description="Accreditation or recognition of the program")


class Recommendations(BaseModel):
    summary: str = Field(..., description="High-level recommendations for improving or positioning the program")
    detailed_points: List[str] = Field(..., description="Specific actionable recommendations")


class UniformOutput(BaseModel):
    program_details: List[ProgramDetails] = Field(..., description="Details about evaluated programs")
    innovations: List[str] = Field(None, description="Emerging trends or innovative approaches identified")
    comparison_summary: str = Field(None, description="Summary of how programs compare against each other")
    recommendations: Recommendations = Field(..., description="Actionable insights and strategies for improvement")
    
@CrewBase
class EduResearchCrew:
    """Crew for analyzing and improving AI education programs"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def target_program_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['target_program_analyzer'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def government_program_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['government_program_researcher'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def traditional_education_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['traditional_education_evaluator'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def online_platform_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['online_platform_researcher'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def coaching_market_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['coaching_market_analyzer'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def corporate_program_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['corporate_program_evaluator'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def vendor_training_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config['vendor_training_evaluator'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def innovation_scout(self) -> Agent:
        return Agent(
            config=self.agents_config['innovation_scout'],
            tools=[YoutubeVideoSearchTool(), YoutubeChannelSearchTool()],
            verbose=True
        )

    @agent
    def program_comparator(self) -> Agent:
        return Agent(
            config=self.agents_config['program_comparator'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @agent
    def strategic_recommendation_developer(self) -> Agent:
        return Agent(
            config=self.agents_config['strategic_recommendation_developer'],
            tools=[SerperDevTool(), WebsiteSearchTool()],
            verbose=True
        )

    @task
    def analyze_target_program_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_target_program_task'],
            agent=self.target_program_analyzer(),
            output_file="analyze_target_program_task.md"
        )

    @task
    def analyze_government_programs(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_government_programs'],
            agent=self.government_program_researcher(),
            output_file="analyze_government_programs.md"
        )

    @task
    def evaluate_traditional_education(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_traditional_education'],
            agent=self.traditional_education_evaluator(),
            output_file="evaluate_traditional_education.md"
        )

    @task
    def assess_online_platforms(self) -> Task:
        return Task(
            config=self.tasks_config['assess_online_platforms'],
            agent=self.online_platform_researcher(),
            output_file="assess_online_platforms.md"
        )

    @task
    def analyze_coaching_market(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_coaching_market'],
            agent=self.coaching_market_analyzer(),
            output_file="analyze_coaching_market.md"
        )

    @task
    def evaluate_corporate_programs(self) -> Task:
        return Task(
            config=self.tasks_config['evaluate_corporate_programs'],
            agent=self.corporate_program_evaluator(),
            output_file="evaluate_corporate_programs.md"
        )

    @task
    def assess_vendor_training(self) -> Task:
        return Task(
            config=self.tasks_config['assess_vendor_training'],
            agent=self.vendor_training_evaluator(),
            output_file="assess_vendor_training.md"
        )

    @task
    def identify_innovations(self) -> Task:
        return Task(
            config=self.tasks_config['identify_innovations'],
            agent=self.innovation_scout(),
            output_file="identify_innovations.md"
        )

    @task
    def compare_programs_task(self) -> Task:
        return Task(
            config=self.tasks_config['compare_programs_task'],
            agent=self.program_comparator(),
            output_file="compare_programs_task.md"
        )

    @task
    def develop_recommendations_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_recommendations_task'],
            agent=self.strategic_recommendation_developer(),
            output_file="develop_recommendations_task.md"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AIProgramResearch crew with sequential execution"""
        return Crew(
            agents=self.agents,
            tasks=[
                self.analyze_target_program_task(),
                self.analyze_government_programs(),
                self.evaluate_traditional_education(),
                self.assess_online_platforms(),
                self.analyze_coaching_market(),
                self.evaluate_corporate_programs(),
                self.assess_vendor_training(),
                self.identify_innovations(),
                self.compare_programs_task(),
                self.develop_recommendations_task()
            ],
            process=Process.sequential,
            verbose=True
        )


