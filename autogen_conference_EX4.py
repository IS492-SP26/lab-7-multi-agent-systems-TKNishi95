"""
AutoGen GroupChat Demo - Interview Platform Product Planning

This demonstrates REAL multi-agent collaboration using AutoGen's GroupChat,
where agents converse with each other, respond to each other's contributions,
and the GroupChatManager orchestrates speaker selection via LLM.

This contrasts with CrewAI's task-based approach — here the agents CHAT
rather than execute isolated tasks.
"""

import os
from datetime import datetime
from config import Config

# Try to import AutoGen
try:
    import autogen
except ImportError:
    print("ERROR: AutoGen is not installed!")
    print("Please run: pip install -r ../requirements.txt")
    exit(1)


class GroupChatInterviewPlatform:
    """Multi-agent GroupChat workflow for interview platform planning using AutoGen"""

    def __init__(self):
        """Initialize the GroupChat with specialized agents"""
        if not Config.validate_setup():
            print("ERROR: Configuration validation failed!")
            exit(1)

        self.config_list = Config.get_config_list()
        self.llm_config = {"config_list": self.config_list, "temperature": Config.AGENT_TEMPERATURE}

        # Create agents and GroupChat
        self._create_agents()
        self._setup_groupchat()

        print("All AutoGen agents created and GroupChat initialized.")

    def _create_agents(self):
        """Create UserProxyAgent and 4 specialist AssistantAgents"""

        # UserProxyAgent acts as the product manager who kicks off the discussion
        self.user_proxy = autogen.UserProxyAgent(
            name="PrincipalOrganizer",
            system_message="A principal organizer who initiates the conference planning discussion and oversees the collaborative process.",
            human_input_mode="NEVER",
            code_execution_config=False,
            max_consecutive_auto_reply=0,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        )

        # Research Agent - starts the conversation with market analysis
        self.research_agent = autogen.AssistantAgent(
            name="SpeakerAgent",
            system_message="""You are a speaker coordinator for a LLM conference held annually in San Francisco. 
            You are responsible for selecting potential speakers (academic, researchers and industry leaders) to give talks at the conference.

Your responsibilities:
- Identify potential speakers for the conference in academia, research and industry.
- Make sure speakers are representative of diverse domain in LLM, including multi-agent research, safety, computer vision etc.
- Identify current trends in these domains to select relevant insightful topics.

When you present your findings, be specific with potential speaker name, topics.
After presenting your research, invite the VenueAgent to find a suitable venue for the conference.
Keep your response focused and under 400 words.""",
            llm_config=self.llm_config,
            description="A speaker coordinator who provides detailed lists of potential invitees to speak on relevant LLM topics.",
        )

        # Analysis Agent - builds on research to identify opportunities
        self.analysis_agent = autogen.AssistantAgent(
            name="VenueAgent",
            system_message="""You are a venue coordinator with expertise in event planning and venue selection.
Your role in this group discussion is to find a suitable venue for the conference.

Your responsibilities:
- Find a suitable venue for the conference in San Francisco, preferably in the downtown area.
- Consider the capacity of the venue, the location, the amenities, the cost, the accessibility, the security, the parking, the food and the drink.
- Consider the size of the conference and the number of attendees.
- Consider the date of the conference and the availability of the venue.
- Consider the budget for the conference and the venue.
- Consider the accessibility of the venue and the parking.
- Consider the security of the venue and the food and the drink.
Keep your response focused and under 400 words. After presenting your venue, invite the ScheduleAgent to create a schedule for the conference.""",
            llm_config=self.llm_config,
            description="A venue coordinator who finds a suitable venue for the conference.",
        )

        # Blueprint Agent - designs the product based on opportunities
        self.blueprint_agent = autogen.AssistantAgent(
            name="ScheduleAgent",
            system_message="""You are an experienced schedule coordinator with expertise in event planning and schedule creation.
Your role in this group discussion is to create a schedule for the conference.

Your responsibilities:
- Create a schedule for the conference, including the talks, the breaks, the lunch, the dinner, the networking event, the keynote speaker, the other speakers.
Keep your response focused and under 400 words. After presenting your schedule, invite the SponsorshipAgent to discuss sponsors for the conference.""",
            llm_config=self.llm_config,
            description="A schedule coordinator who creates a schedule for the conference.",
        )

        # Reviewer Agent - reviews and concludes with strategic recommendations
        self.reviewer_agent = autogen.AssistantAgent(
            name="SponsorshipAgent",
            system_message="""You are a sponsorship coordinator with expertise in event planning and sponsorship acquisition.
Your role in this group discussion is to acquire sponsors for the conference.

Your responsibilities:
- Acquire sponsors for the conference, including the platinum, gold, silver, bronze sponsors.

Keep your response focused and under 400 words. After presenting your sponsors, conclude the discussion by ending your message with the word TERMINATE.""",
            llm_config=self.llm_config,
            description="A sponsorship coordinator who acquires sponsors for the conference.",
        )

    def _setup_groupchat(self):
        """Create the GroupChat and GroupChatManager"""
        self.groupchat = autogen.GroupChat(
            agents=[
                self.user_proxy,
                self.research_agent,
                self.analysis_agent,
                self.blueprint_agent,
                self.reviewer_agent,
            ],
            messages=[],
            max_round=10,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
            send_introductions=True,
        )

        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=self.llm_config,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
        )

    def run(self):
        """Execute the GroupChat workflow"""
        print("\n" + "=" * 80)
        print("AUTOGEN GROUPCHAT - AI INTERVIEW PLATFORM PRODUCT PLANNING")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {Config.OPENAI_MODEL}")
        print(f"Max Rounds: {self.groupchat.max_round}")
        print(f"Speaker Selection: {self.groupchat.speaker_selection_method}")
        print("\nAgents in GroupChat:")
        for agent in self.groupchat.agents:
            print(f"  - {agent.name}")
        print("\n" + "=" * 80)
        print("MULTI-AGENT CONVERSATION BEGINS")
        print("=" * 80 + "\n")

        # Initiate the group chat conversation
        initial_message = """Team, we need to develop a product plan for an AI-powered interview platform.

Let's collaborate on this:
1. ResearchAgent: Start by analyzing the competitive landscape
2. AnalysisAgent: Then identify key market opportunities
3. BlueprintAgent: Design the product features and user journey
4. ReviewerAgent: Finally, review and provide strategic recommendations

ResearchAgent, please begin with your market analysis."""

        chat_result = self.user_proxy.initiate_chat(
            self.manager,
            message=initial_message,
            summary_method="reflection_with_llm",
            summary_args={
                "summary_prompt": "Summarize the complete product plan developed through this multi-agent discussion. Include: key market findings, identified opportunities, proposed features, and strategic recommendations."
            },
        )

        # Print results
        self._print_summary(chat_result)

        # Save to file
        output_file = self._save_results(chat_result)
        print(f"\nFull results saved to: {output_file}")

        print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def _print_summary(self, chat_result):
        """Print educational summary highlighting GroupChat behavior"""
        print("\n" + "=" * 80)
        print("CONVERSATION COMPLETE")
        print("=" * 80)

        print(f"\nTotal conversation rounds: {len(self.groupchat.messages)}")
        print("\nSpeaker order (as selected by GroupChatManager):")
        for i, msg in enumerate(self.groupchat.messages, 1):
            speaker = msg.get("name", "Unknown")
            content = msg.get("content", "")
            preview = content[:80].replace("\n", " ") + "..." if len(content) > 80 else content.replace("\n", " ")
            print(f"  {i}. [{speaker}]: {preview}")

        if chat_result.summary:
            print("\n" + "-" * 80)
            print("EXECUTIVE SUMMARY (LLM-generated reflection)")
            print("-" * 80)
            print(chat_result.summary)

        print("\n" + "-" * 80)
        print("EDUCATIONAL NOTE: AutoGen vs CrewAI")
        print("-" * 80)
        print("""
This workflow demonstrated AutoGen's CONVERSATIONAL approach to multi-agent systems:
- Agents were placed in a GroupChat and communicated naturally
- The GroupChatManager used LLM-based speaker selection (not hardcoded order)
- Agents referenced each other's contributions in their responses
- The conversation emerged organically through agent-to-agent interaction

Compare with CrewAI (crewai/crewai_demo.py):
- CrewAI assigns discrete Tasks to Agents with expected_output
- Each agent works independently on their assigned task
- Output is passed as context to the next task (not conversational)
- Workflow is strictly sequential with no back-and-forth
""")

    def _save_results(self, chat_result):
        """Save GroupChat conversation and summary to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(output_dir, f"groupchat_output_{timestamp}.txt")

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AUTOGEN GROUPCHAT - AI INTERVIEW PLATFORM PRODUCT PLAN\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {Config.OPENAI_MODEL}\n")
            f.write(f"Conversation Rounds: {len(self.groupchat.messages)}\n\n")

            f.write("=" * 80 + "\n")
            f.write("MULTI-AGENT CONVERSATION\n")
            f.write("=" * 80 + "\n\n")

            for i, msg in enumerate(self.groupchat.messages, 1):
                speaker = msg.get("name", "Unknown")
                content = msg.get("content", "")
                f.write(f"--- Turn {i}: {speaker} ---\n")
                f.write(content + "\n\n")

            if chat_result.summary:
                f.write("=" * 80 + "\n")
                f.write("EXECUTIVE SUMMARY\n")
                f.write("=" * 80 + "\n")
                f.write(chat_result.summary + "\n")

        return output_file


if __name__ == "__main__":
    try:
        workflow = GroupChatInterviewPlatform()
        workflow.run()
        print("\nGroupChat workflow completed successfully!")
    except Exception as e:
        print(f"\nError during workflow execution: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Verify API key is set in parent directory .env (../.env)")
        print("2. Check your API key has sufficient credits")
        print("3. Ensure pyautogen is installed: pip install -r ../requirements.txt")
        print("4. Verify internet connection")
        import traceback
        traceback.print_exc()
