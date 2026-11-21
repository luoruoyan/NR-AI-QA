"""Accept coverage recommendations without granting the model write access to coverage."""

from pydantic import ValidationError

from nr_ai_qa.ai.schemas import Advice

REASONS = {
    "boundary": "Exercise a documented boundary with independent checks.",
    "interaction": "Cover a supported combination of parameters tested separately.",
    "missing_evidence": "Collect actual execution evidence for the remaining scope.",
    "reproduce_history": "Reproduce a known observation with a fixed seed and configuration.",
    "seed_variation": "Repeat a verified configuration with an independent payload seed.",
}


def apply_advice(payload: dict, registry, family: str) -> dict:
    items = payload.get("advice")
    if set(payload) != {"advice"} or not isinstance(items, list) or len(items) > 12:
        raise ValueError("Invalid advice response envelope")
    accepted, rejected, seen = [], [], set()
    for index, item in enumerate(items):
        try:
            proposal = Advice.model_validate(item)
            topic = registry.config["topics"].get(proposal.topic_id)
            if (not topic or topic["family"] != family or topic.get("state") == "UNSUPPORTED"
                    or not set(proposal.requirement_ids) <= set(topic["requirement_ids"])):
                raise ValueError("Unknown or out-of-scope topic or requirement")
            if proposal.topic_id in seen:
                raise ValueError("Duplicate topic recommendation")
        except (ValidationError, ValueError) as exc:
            rejected.append({"index": index, "reason": str(exc).splitlines()[0]})
            continue
        seen.add(proposal.topic_id)
        accepted.append(proposal.model_dump() | {"explanation": REASONS[proposal.reason]})
    return {"accepted": accepted, "rejected": rejected}
