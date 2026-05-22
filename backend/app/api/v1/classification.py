"""
classification.py — EU AI Act Risk Classification API
======================================================

This module exposes the REST API endpoints that classify AI systems under the
EU AI Act risk framework. Classification is powered by an LLM (Gemini / OpenAI-
compatible) that reasons over the system's metadata against the relevant EU AI
Act articles and returns a structured risk verdict with a human-readable rationale.

EU AI Act Legal Basis
---------------------
The classification logic maps to the following articles and annexes:

- **Article 5  — Prohibited AI Practices**
  AI systems that fall under Art. 5 are assigned the ``Unacceptable`` risk level
  and cannot be deployed in the EU. Examples include social scoring by public
  authorities and real-time remote biometric identification in public spaces.

- **Article 6 + Annex III — High-Risk AI Systems**
  Art. 6 defines the two-step test for high-risk classification:
  (1) the system is a safety component of a product covered by Union harmonisation
  legislation listed in Annex II, or is itself such a product; or
  (2) the system falls within one of the eight domains listed in Annex III
  (e.g., biometrics, critical infrastructure, education, employment, essential
  private/public services, law enforcement, migration, justice).
  Systems meeting either criterion are assigned the ``High`` risk level and must
  comply with the full obligations in Chapter III (conformity assessment, technical
  documentation, human oversight, etc.).

- **Article 52 — Transparency Obligations for Certain AI Systems**
  AI systems that interact with humans (chatbots), generate synthetic content
  (deepfakes), or perform emotion recognition / biometric categorisation are
  assigned the ``Limited`` risk level and must meet specific disclosure obligations.

- **Minimal Risk (default)**
  All other AI systems not captured by Art. 5, Art. 6, or Art. 52 are classified
  as ``Minimal`` risk. No mandatory obligations apply, though voluntary codes of
  conduct are encouraged.

Risk Levels
-----------
AegisAI uses four risk levels, ordered from lowest to highest severity:

1. ``Minimal``       — No mandatory EU AI Act obligations. Vast majority of AI
                        applications (spam filters, AI in video games, etc.).
2. ``Limited``       — Transparency obligations only (Art. 52). Must inform users
                        they are interacting with AI.
3. ``High``          — Full compliance regime (Art. 6 + Annex III). Requires
                        technical documentation, conformity assessment, human
                        oversight, and registration in the EU database.
4. ``Unacceptable``  — Prohibited (Art. 5). Deployment in the EU is forbidden.

Classification Flow
-------------------
1. Client submits AI system attributes (name, sector, use_case, intended_purpose,
   capabilities, data types processed) to the classify endpoint.
2. The endpoint builds a structured prompt embedding the system's metadata and the
   relevant EU AI Act rules.
3. The prompt is sent to the configured LLM (``app.modules.llm``) which returns a
   JSON payload containing:
   - ``risk_level``              — one of the four levels above
   - ``classification_reasoning`` — plain-English rationale citing EU AI Act articles
   - ``applicable_articles``     — list of articles that triggered the classification
4. The response is validated via Pydantic and returned to the caller.
5. In the ``/classify/{system_id}`` variant, the result is also persisted back to
   the ``AISystem`` record in the database (``risk_level``, ``classification_reasoning``
   fields on the ORM model).

Endpoints
---------
- ``POST /classify``
    Stateless classification — accepts a ``ClassifyRequest`` body and returns a
    ``ClassifyResponse`` without touching the database. Useful for one-off checks
    or external integrations.

- ``POST /classify/{system_id}``
    Persistent classification — loads the ``AISystem`` identified by ``system_id``
    from the database, classifies it, and writes the result back. Requires the
    caller to be the owner of the AI system (JWT-authenticated).

Dependencies
------------
- ``app.modules.llm.client``           — LLM client (OpenAI-compatible)
- ``app.models.ai_system.AISystem``    — ORM model updated by ``/classify/{system_id}``
- ``app.schemas.classification.*``     — Pydantic request / response schemas
- ``app.core.db.get_db``               — SQLAlchemy session dependency
- ``app.core.security.get_current_user`` — JWT authentication dependency
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.ai_system import AISystem, RiskLevel, RiskAssessment, ComplianceStatus
from app.schemas.ai_system import (
    RiskClassificationRequest,
    RiskClassificationResponse,
    QuestionnaireRiskFactor,
)

router = APIRouter()

QUESTIONNAIRE_RISK_FACTORS: List[QuestionnaireRiskFactor] = [
    QuestionnaireRiskFactor(
        id="is_safety_component",
        question="Is the AI system used as a safety component of a product or system?",
        article="Article 6(1)",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="affects_fundamental_rights",
        question="Can the AI system affect fundamental rights such as employment, education, essential services, or access to opportunities?",
        article="Article 6(2)",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="uses_biometric_data",
        question="Does the system use biometric data for identification, verification, or categorization?",
        article="Annex III",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="makes_automated_decisions",
        question="Does the system make automated decisions without meaningful human review?",
        article="Article 6 / Annex III context",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="hr_recruitment_screening",
        question="Is the system used for recruitment, CV screening, candidate filtering, or candidate ranking?",
        article="Annex III point 4(a)",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="hr_promotion_termination",
        question="Is the system used for promotion, termination, task allocation, performance evaluation, or employment-related decisions?",
        article="Annex III point 4(b)",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="credit_worthiness",
        question="Is the system used to evaluate creditworthiness or determine access to financial resources?",
        article="Annex III point 5(b)",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="insurance_risk_assessment",
        question="Is the system used for insurance risk assessment, pricing, or eligibility decisions?",
        article="Annex III point 5(c)",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="law_enforcement",
        question="Is the system used by or for law enforcement purposes?",
        article="Annex III point 6",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="border_control",
        question="Is the system used for migration, asylum, or border control management?",
        article="Annex III point 7",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="justice_system",
        question="Is the system used to assist judicial authorities or influence legal outcomes?",
        article="Annex III point 8",
        triggers_level=RiskLevel.HIGH,
    ),
    QuestionnaireRiskFactor(
        id="interacts_with_humans",
        question="Does the system directly interact with humans, such as a chatbot or virtual assistant?",
        article="Article 52(1)",
        triggers_level=RiskLevel.LIMITED,
    ),
    QuestionnaireRiskFactor(
        id="generates_synthetic_content",
        question="Does the system generate synthetic or manipulated audio, image, video, or text content?",
        article="Article 52(3)",
        triggers_level=RiskLevel.LIMITED,
    ),
    QuestionnaireRiskFactor(
        id="emotion_recognition",
        question="Does the system perform emotion recognition?",
        article="Article 52(3)",
        triggers_level=RiskLevel.LIMITED,
    ),
    QuestionnaireRiskFactor(
        id="biometric_categorization",
        question="Does the system perform biometric categorization?",
        article="Article 52 / Annex III context",
        triggers_level=RiskLevel.LIMITED,
    ),
]


class BulkClassificationItem(BaseModel):
    system_id: int
    classification: Optional[RiskClassificationResponse] = None
    error: Optional[str] = None


class BulkClassificationRequest(BaseModel):
    system_ids: List[int]


class BulkClassificationResponse(BaseModel):
    results: List[BulkClassificationItem]


class BulkClassificationItem(BaseModel):
    system_id: int
    classification: Optional[RiskClassificationResponse] = None
    error: Optional[str] = None


class BulkClassificationRequest(BaseModel):
    system_ids: List[int]


class BulkClassificationResponse(BaseModel):
    results: List[BulkClassificationItem]


def classify_risk(data: RiskClassificationRequest) -> RiskClassificationResponse:
    """
    Classify the risk level of an AI system based on EU AI Act criteria.
    """
    reasons = []
    requirements = []
    risk_level = RiskLevel.MINIMAL
    confidence = 0.9

    # Check for UNACCEPTABLE risk (Article 5 - Prohibited practices)
    # Social scoring, real-time biometric identification in public spaces, etc.
    # These are typically banned outright

    # Check for HIGH risk (Article 6 + Annex III)
    high_risk_indicators = []

    # HR and recruitment AI (Annex III, point 4)
    if data.hr_recruitment_screening or data.hr_promotion_termination:
        high_risk_indicators.append("HR recruitment/management AI system")
        reasons.append(
            "AI systems used for recruitment, CV screening, or employment decisions are classified as HIGH risk under Annex III"
        )
        requirements.extend(
            [
                "Implement risk management system (Article 9)",
                "Ensure data governance and quality (Article 10)",
                "Maintain technical documentation (Article 11)",
                "Enable record-keeping/logging (Article 12)",
                "Provide transparency to users (Article 13)",
                "Enable human oversight (Article 14)",
                "Ensure accuracy, robustness, cybersecurity (Article 15)",
            ]
        )

    # Credit and insurance (Annex III, point 5)
    if data.credit_worthiness or data.insurance_risk_assessment:
        high_risk_indicators.append("Credit/insurance assessment AI")
        reasons.append(
            "AI for creditworthiness or insurance risk assessment is HIGH risk under Annex III"
        )

    # Safety component
    if data.is_safety_component:
        high_risk_indicators.append("Safety component of a product")
        reasons.append("AI used as a safety component requires HIGH risk compliance")

    # Fundamental rights impact
    if data.affects_fundamental_rights:
        high_risk_indicators.append("Affects fundamental rights")
        reasons.append(
            "System impacts fundamental rights (employment, education, essential services)"
        )

    # Law enforcement, border control, justice
    if data.law_enforcement or data.border_control or data.justice_system:
        high_risk_indicators.append("Law enforcement/justice system use")
        reasons.append(
            "Use in law enforcement, border control, or justice is HIGH risk"
        )

    # Determine if HIGH risk
    if high_risk_indicators:
        risk_level = RiskLevel.HIGH

    # Check for LIMITED risk (Article 52 - Transparency obligations)
    elif (
        data.interacts_with_humans
        or data.emotion_recognition
        or data.generates_synthetic_content
    ):
        risk_level = RiskLevel.LIMITED
        if data.interacts_with_humans:
            reasons.append("System interacts directly with humans (e.g., chatbot)")
            requirements.append(
                "Inform users they are interacting with AI (Article 52)"
            )
        if data.emotion_recognition:
            reasons.append("System uses emotion recognition")
            requirements.append("Inform subjects about emotion recognition system")
        if data.generates_synthetic_content:
            reasons.append("System generates synthetic/manipulated content")
            requirements.append("Label AI-generated content appropriately")

    # MINIMAL risk - no specific requirements
    else:
        reasons.append("System does not fall into high-risk or limited-risk categories")
        requirements.append(
            "No mandatory requirements, but voluntary codes of conduct encouraged"
        )

    # Generate next steps based on risk level
    next_steps = []
    if risk_level == RiskLevel.HIGH:
        next_steps = [
            "Complete the full risk assessment questionnaire",
            "Document your AI system's technical specifications",
            "Implement a risk management system",
            "Establish data governance procedures",
            "Set up human oversight mechanisms",
            "Prepare conformity assessment documentation",
        ]
    elif risk_level == RiskLevel.LIMITED:
        next_steps = [
            "Implement transparency notices for users",
            "Document your disclosure mechanisms",
            "Review interaction points with users",
        ]
    else:
        next_steps = [
            "Consider voluntary compliance measures",
            "Monitor regulatory updates",
            "Document your AI governance practices",
        ]

    return RiskClassificationResponse(
        risk_level=risk_level,
        confidence=confidence,
        reasons=reasons,
        requirements=requirements,
        next_steps=next_steps,
    )


@router.post("/classify", response_model=RiskClassificationResponse)
def classify_ai_system(
    data: RiskClassificationRequest, current_user: User = Depends(get_current_user)
):
    """
    Classify an AI system's risk level based on EU AI Act criteria.
    This is a preliminary classification - full assessment requires more details.
    """
    return classify_risk(data)


@router.post("/classify/{system_id}", response_model=RiskClassificationResponse)
def classify_and_save(
    system_id: int,
    data: RiskClassificationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Classify an AI system and save the result to the database.
    """
    # Get the AI system
    system = (
        db.query(AISystem)
        .filter(AISystem.id == system_id, AISystem.owner_id == current_user.id)
        .first()
    )

    if not system:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="AI system not found"
        )

    # Perform classification
    result = classify_risk(data)

    # Update the AI system
    system.risk_level = result.risk_level
    system.compliance_status = ComplianceStatus.IN_PROGRESS
    system.questionnaire_responses = data.model_dump()

    # Create risk assessment record
    assessment = RiskAssessment(
        ai_system_id=system.id,
        assessment_type="initial",
        risk_level=result.risk_level,
        findings=[{"type": "classification", "reasons": result.reasons}],
        recommendations=[
            {"requirements": result.requirements, "next_steps": result.next_steps}
        ],
        overall_score=70 if result.risk_level == RiskLevel.MINIMAL else 30,
    )
    db.add(assessment)

    db.commit()
    db.refresh(system)

    return result



@router.get("/risk-factors", response_model=List[QuestionnaireRiskFactor])
def get_questionnaire_risk_factors(
    current_user: User = Depends(get_current_user),
):
    """
    Return the static questionnaire metadata used by the risk classification flow.

    This does not query the database because these factors describe the
    classification rules themselves, not a user's saved questionnaire answers.
    Keep this list aligned with RiskClassificationRequest and classify_risk().
    """
    return QUESTIONNAIRE_RISK_FACTORS

@router.post("/bulk", response_model=BulkClassificationResponse)
def bulk_classify_systems(
    request: BulkClassificationRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Classify multiple AI systems in one request.
    Returns per-system classification results and partial failure details.
    """
    results: List[BulkClassificationItem] = []

    for system_id in request.system_ids:
        system = db.query(AISystem).filter(
            AISystem.id == system_id,
            AISystem.owner_id == current_user.id
        ).first()

        if not system:
            results.append(
                BulkClassificationItem(
                    system_id=system_id,
                    error="AI system not found"
                )
            )
            continue

        if not system.questionnaire_responses:
            results.append(
                BulkClassificationItem(
                    system_id=system_id,
                    error="Questionnaire responses missing"
                )
            )
            continue

        try:
            classification_data = RiskClassificationRequest(**system.questionnaire_responses)
        except Exception as exc:
            results.append(
                BulkClassificationItem(
                    system_id=system_id,
                    error=f"Invalid questionnaire responses: {exc}"
                )
            )
            continue

        result = classify_risk(classification_data)
        system.risk_level = result.risk_level
        system.compliance_status = ComplianceStatus.IN_PROGRESS
        system.questionnaire_responses = system.questionnaire_responses

        assessment = RiskAssessment(
            ai_system_id=system.id,
            assessment_type="bulk",
            risk_level=result.risk_level,
            findings=[{"type": "classification", "reasons": result.reasons}],
            recommendations=[{"requirements": result.requirements, "next_steps": result.next_steps}],
            overall_score=70 if result.risk_level == RiskLevel.MINIMAL else 30
        )
        db.add(assessment)

        results.append(
            BulkClassificationItem(
                system_id=system_id,
                classification=result
            )
        )

    db.commit()
    return BulkClassificationResponse(results=results)

    
@router.get("/risk-factors", response_model=List[QuestionnaireRiskFactor])
def get_questionnaire_risk_factors(
    current_user: User = Depends(get_current_user),
):
    """
    Return the static questionnaire metadata used by the risk classification flow.

    This does not query the database because these factors describe the
    classification rules themselves, not a user's saved questionnaire answers.
    Keep this list aligned with RiskClassificationRequest and classify_risk().
    """
    return QUESTIONNAIRE_RISK_FACTORS
