from string import Template
from typing import Optional

from sqlalchemy.orm import Session

from model import ExpertPipelineStepOutput, ExpertPipelineMessage


class LLMContextManager:
    def __init__(
        self,
        user_id,
        session: Session,
        business_context_preset_id: Optional[str] = None,
    ):
        self.db = session
        self.user_id = user_id
        self.business_context_preset_id = business_context_preset_id

    def get_user_instructions(self) -> str:
        prompt_template = Template(
            "IMPORTANT FOLLOW THESE INSTRUCTIONS WHEN ANSWERING TO ME: ${instructions}"
        )
        return prompt_template.substitute(
            instructions="Be precise and concise in your answers."
        )

    def get_chat_history(self, thread_id: str, depth=0) -> list[(str, str)]:
        depth = 100 if depth <= 0 else depth
        messages = (
            self.db.query(ExpertPipelineMessage)
            .filter_by(thread_id=thread_id, is_archived=False)
            .order_by(ExpertPipelineMessage.time_created.desc())
            .limit(depth)
            .all()
        )
        messages.reverse()
        history = []
        for message in messages:
            step_outs = (
                self.db.query(ExpertPipelineStepOutput)
                .filter_by(message_id=message.id)
                .order_by(ExpertPipelineStepOutput.time_created.asc())
                .all()
            )

            history.append(("user", message.query))
            for step_out in step_outs:
                ai = step_out.chat_history()
                if ai != "":
                    history.append(("ai", ai))
        return history
