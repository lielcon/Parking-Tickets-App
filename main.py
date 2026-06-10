from datetime import datetime, timedelta
import os
from pathlib import Path
import re

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import vision
from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

app = FastAPI(title="Parking Tickets API")

# CORS: allows browser requests from Expo web / frontend during development.
# CORSMiddleware answers OPTIONS preflight requests and adds CORS headers.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load variables from a local .env file into os.environ.
# This keeps secrets like DB credentials out of the code.
load_dotenv()

# Read the PostgreSQL connection string from .env.
# Example:
# DATABASE_URL=postgresql://username:password@localhost:5432/parking_db
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing. Add it to your .env file.")

# Create the SQLAlchemy engine for PostgreSQL.
# No SQLite-specific connect args are needed here.
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# SQLAlchemy ORM model: this maps to the "users" table in the database.
# We keep this very simple for now (id, email, password) to support frontend login flow.
class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)


# SQLAlchemy ORM model: this maps to the "tickets" table in the database.
# A ticket can optionally belong to a user (user_id) for user-specific views.
class TicketDB(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    city = Column(String, nullable=False)
    plate_number = Column(String, nullable=False, index=True)
    ticket_number = Column(String, nullable=False)
    issued_at = Column(DateTime, nullable=False)
    payable_at = Column(DateTime, nullable=False)
    status = Column(String, nullable=False)
    fine_amount = Column(String, nullable=True)


# SQLAlchemy ORM model: this maps to the existing "lawyers" table.
# We only describe the columns here; we do not change the table structure.
class LawyerDB(Base):
    __tablename__ = "lawyers"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    city = Column(String, nullable=False, index=True)
    specialty = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)


# SQLAlchemy ORM model: this maps to the "appeal_requests" table.
# An appeal request links a user, a ticket, and a lawyer together.
class AppealRequestDB(Base):
    __tablename__ = "appeal_requests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.id"), nullable=False, index=True)
    lawyer_id = Column(Integer, ForeignKey("lawyers.id"), nullable=False, index=True)
    message = Column(String, nullable=False)
    status = Column(String, nullable=False, default="pending")
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# Create DB tables if they do not exist yet.
Base.metadata.create_all(bind=engine)


# FastAPI dependency: gives each request its own DB session.
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class TicketCreate(BaseModel):
    city: str
    plate_number: str
    ticket_number: str
    issued_at: datetime


class Ticket(BaseModel):
    id: int
    user_id: int | None = None
    city: str
    plate_number: str
    ticket_number: str
    issued_at: datetime
    payable_at: datetime
    status: str
    is_payable: bool = False
    fine_amount: str | None = None


class Lawyer(BaseModel):
    id: int
    full_name: str
    phone: str | None = None
    email: str | None = None
    city: str
    specialty: str | None = None
    is_active: bool

    # Allow building this response directly from a SQLAlchemy LawyerDB object.
    model_config = {"from_attributes": True}


class AppealRequestCreate(BaseModel):
    user_id: int
    ticket_id: int
    lawyer_id: int
    message: str


class AppealRequest(BaseModel):
    id: int
    user_id: int
    ticket_id: int
    lawyer_id: int
    message: str
    status: str
    created_at: datetime

    # Allow building this response directly from a SQLAlchemy AppealRequestDB object.
    model_config = {"from_attributes": True}


class UserRegisterRequest(BaseModel):
    email: str
    password: str


class UserLoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str


class ManualTicketCreate(BaseModel):
    user_id: int
    ticket_number: str
    plate_number: str
    city: str
    issued_date: str


uploads_dir = Path("uploads")


def to_ticket_schema(ticket_db: TicketDB) -> Ticket:
    """Convert SQLAlchemy ticket object to API response schema."""
    return Ticket(
        id=ticket_db.id,
        user_id=ticket_db.user_id,
        city=ticket_db.city,
        plate_number=ticket_db.plate_number,
        ticket_number=ticket_db.ticket_number,
        issued_at=ticket_db.issued_at,
        payable_at=ticket_db.payable_at,
        status=ticket_db.status,
        fine_amount=ticket_db.fine_amount,
    )


def get_status_for_response(payable_at: datetime, current_time: datetime) -> str:
    """
    Decide status only for API response (no DB write):
    - before payable_at  -> not_payable
    - at/after payable_at -> payable
    """
    if current_time < payable_at:
        return "not_payable"
    return "payable"


def get_is_payable_for_response(payable_at: datetime, current_time: datetime) -> bool:
    """
    Return True only when ticket can be paid now.
    - current_time >= payable_at -> True
    - current_time < payable_at  -> False
    """
    return current_time >= payable_at


async def extract_ocr_text_from_file(file: UploadFile) -> str:
    """Read uploaded image and return full OCR text."""
    # 1) Read image bytes from upload.
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 2) Send image to Google Vision OCR.
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.document_text_detection(image=image)

    if response.error.message:
        raise HTTPException(status_code=500, detail=response.error.message)

    # 3) Return the full OCR text (or empty string if none).
    return response.full_text_annotation.text or ""


def extract_by_label(text: str, label: str) -> str | None:
    """
    Find a value after a known label.
    Example pattern: 'מועד העבירה: 01/01/2026 12:30'
    """
    pattern = rf"{re.escape(label)}\s*[:\-]?\s*(.+)"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1).strip()


def parse_ticket_fields(text: str) -> dict[str, str | None]:
    """Parse key ticket fields from OCR text using simple rules."""
    # City: simple keyword check.
    city = None
    if "תל אביב" in text:
        city = "תל אביב"
    elif "Tel Aviv" in text:
        city = "Tel Aviv"

    # Other fields: find values after known Hebrew labels.
    return {
        "city": city,
        "plate_number": extract_by_label(text, "מספר כלי הרכב"),
        "ticket_number": extract_by_label(text, "הודעת תשלום קנס מספר"),
        "issued_at": extract_by_label(text, "מועד העבירה"),
        "fine_amount": extract_by_label(text, "סכום הקנס"),
    }


def parse_issued_at_datetime(value: str | None) -> datetime | None:
    """Convert parsed issued_at text into datetime."""
    if not value:
        return None

    # Keep this simple: try a few common formats.
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    # Also allow ISO values like 2026-04-06T10:30:00
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@app.get("/")
def read_root():
    return {"message": "Welcome to the Parking Tickets API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/tickets", response_model=Ticket)
def create_ticket(ticket_data: TicketCreate, db: Session = Depends(get_db)):
    # Save a new ticket row in the database.
    new_ticket = TicketDB(
        city=ticket_data.city,
        plate_number=ticket_data.plate_number,
        ticket_number=ticket_data.ticket_number,
        issued_at=ticket_data.issued_at,
        payable_at=ticket_data.issued_at + timedelta(hours=48),
        status="pending",
    )
    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)
    return to_ticket_schema(new_ticket)


@app.post("/register", response_model=UserResponse)
def register_user(user_data: UserRegisterRequest, db: Session = Depends(get_db)):
    # Basic register flow:
    # 1) check if email already exists
    # 2) create user
    # 3) return only safe fields (id + email)
    existing_user = db.query(UserDB).filter(UserDB.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="User with this email already exists.")

    new_user = UserDB(email=user_data.email, password=user_data.password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserResponse(id=new_user.id, email=new_user.email)


@app.post("/login", response_model=UserResponse)
def login_user(login_data: UserLoginRequest, db: Session = Depends(get_db)):
    # Basic login flow:
    # find user by email + password, otherwise return 401.
    user = (
        db.query(UserDB)
        .filter(UserDB.email == login_data.email, UserDB.password == login_data.password)
        .first()
    )
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return UserResponse(id=user.id, email=user.email)


@app.post("/manual-ticket", response_model=Ticket)
def create_manual_ticket(ticket_data: ManualTicketCreate, db: Session = Depends(get_db)):
    # Manual ticket flow:
    # frontend sends known values including issued_date (YYYY-MM-DD).
    # We convert it to issued_at at 00:00, then compute payable_at (+48h).
    user = db.query(UserDB).filter(UserDB.id == ticket_data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    try:
        issued_at = datetime.strptime(ticket_data.issued_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="issued_date must be in YYYY-MM-DD format.")

    payable_at = issued_at + timedelta(hours=48)
    now = datetime.utcnow()
    status = "payable" if now >= payable_at else "not_payable"

    # Block duplicates: same ticket_number for the same user.
    existing_ticket = (
        db.query(TicketDB)
        .filter(
            TicketDB.user_id == ticket_data.user_id,
            TicketDB.ticket_number == ticket_data.ticket_number,
        )
        .first()
    )
    if existing_ticket:
        raise HTTPException(status_code=409, detail="This ticket was already added.")

    new_ticket = TicketDB(
        user_id=ticket_data.user_id,
        city=ticket_data.city,
        plate_number=ticket_data.plate_number,
        ticket_number=ticket_data.ticket_number,
        issued_at=issued_at,
        payable_at=payable_at,
        status=status,
    )
    db.add(new_ticket)
    db.commit()
    db.refresh(new_ticket)
    return to_ticket_schema(new_ticket)


@app.get("/tickets", response_model=list[Ticket])
def get_tickets(
    user_id: int | None = None,
    plate_number: str | None = None,
    status: str | None = None,
    db: Session = Depends(get_db),
):
    # Read tickets from DB. Optionally filter by plate number.
    # We calculate status dynamically only when building the response.
    # This keeps database storage unchanged.
    # New filters allow frontend to fetch per-user and per-status ticket lists.
    now = datetime.utcnow()
    query = db.query(TicketDB)
    if user_id is not None:
        query = query.filter(TicketDB.user_id == user_id)
    if plate_number is not None:
        query = query.filter(TicketDB.plate_number == plate_number)
    if status is not None:
        query = query.filter(TicketDB.status == status)

    db_tickets = query.all()
    response_tickets = [to_ticket_schema(ticket) for ticket in db_tickets]
    for ticket in response_tickets:
        # Keep "paid" as-is. Otherwise, status is still computed from payable_at.
        if ticket.status != "paid":
            ticket.status = get_status_for_response(ticket.payable_at, now)
        ticket.is_payable = get_is_payable_for_response(ticket.payable_at, now)
    return response_tickets


@app.post("/appeal-requests", response_model=AppealRequest)
def create_appeal_request(payload: AppealRequestCreate, db: Session = Depends(get_db)):
    # 1) Basic message validation.
    if len(payload.message.strip()) < 20:
        raise HTTPException(status_code=400, detail="message must be at least 20 characters.")

    # 2) Validate referenced user.
    user = db.query(UserDB).filter(UserDB.id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # 3) Validate referenced ticket.
    ticket = db.query(TicketDB).filter(TicketDB.id == payload.ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")

    # 4) Validate referenced lawyer and active status.
    lawyer = db.query(LawyerDB).filter(LawyerDB.id == payload.lawyer_id).first()
    if not lawyer:
        raise HTTPException(status_code=404, detail="Lawyer not found.")
    if not lawyer.is_active:
        raise HTTPException(status_code=400, detail="Lawyer is not active.")

    # 5) Save new appeal request. Status defaults to "pending".
    appeal_request = AppealRequestDB(
        user_id=payload.user_id,
        ticket_id=payload.ticket_id,
        lawyer_id=payload.lawyer_id,
        message=payload.message.strip(),
        status="pending",
        created_at=datetime.utcnow(),
    )
    db.add(appeal_request)
    db.commit()
    db.refresh(appeal_request)
    return appeal_request


@app.get("/appeal-requests", response_model=list[AppealRequest])
def get_appeal_requests(user_id: int, db: Session = Depends(get_db)):
    # Return all appeal requests for this user.
    requests = (
        db.query(AppealRequestDB)
        .filter(AppealRequestDB.user_id == user_id)
        .order_by(AppealRequestDB.created_at.desc())
        .all()
    )
    return requests


@app.get("/lawyers", response_model=list[Lawyer])
def get_lawyers(city: str, db: Session = Depends(get_db)):
    # Return active lawyers for an exact city match.
    # If none are found, this naturally returns an empty list [].
    lawyers = (
        db.query(LawyerDB)
        .filter(LawyerDB.city == city, LawyerDB.is_active == True)  # noqa: E712
        .all()
    )
    return lawyers


@app.patch("/tickets/{ticket_id}/mark-paid", response_model=Ticket)
def mark_ticket_paid(ticket_id: int, db: Session = Depends(get_db)):
    # Payment flow: update one ticket to paid and return updated record.
    ticket = db.query(TicketDB).filter(TicketDB.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found.")
    ticket.status = "paid"
    db.commit()
    db.refresh(ticket)
    return to_ticket_schema(ticket)


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    uploads_dir.mkdir(exist_ok=True)
    file_path = uploads_dir / file.filename

    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    return {"filename": file.filename, "path": str(file_path)}


@app.post("/ocr")
async def extract_text_with_ocr(file: UploadFile = File(...)):
    try:
        extracted_text = await extract_ocr_text_from_file(file)
        return {"text": extracted_text}

    except HTTPException:
        raise
    except Exception as exc:
        # Keep error response clean for API users.
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc


@app.post("/parse-ticket")
async def parse_ticket_data(file: UploadFile = File(...)):
    try:
        # Step 1: Extract raw text from the uploaded image using OCR.
        text = await extract_ocr_text_from_file(file)

        # Step 2: Parse a few known fields from the OCR text.
        return parse_ticket_fields(text)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ticket parsing failed: {exc}") from exc


@app.post("/scan-ticket", response_model=Ticket)
async def scan_ticket(
    file: UploadFile = File(...),
    user_id: int | None = Form(None),
    db: Session = Depends(get_db),
):
    try:
        # Step 1: OCR the image and parse structured ticket fields.
        text = await extract_ocr_text_from_file(file)
        parsed = parse_ticket_fields(text)

        # Step 2: Validate fields needed to create a Ticket object.
        issued_at_dt = parse_issued_at_datetime(parsed["issued_at"])
        if not parsed["city"] or not parsed["plate_number"] or not parsed["ticket_number"] or not issued_at_dt:
            raise HTTPException(
                status_code=400,
                detail="Could not parse required ticket fields (city, plate_number, ticket_number, issued_at).",
            )

        # Step 3: If user_id was sent, link this scanned ticket to that user.
        # If no user_id was sent, keep old behavior and save without a user.
        if user_id is not None:
            user = db.query(UserDB).filter(UserDB.id == user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found.")

        # Step 3b: Block duplicates: same ticket_number for the same user.
        existing_ticket = (
            db.query(TicketDB)
            .filter(
                TicketDB.user_id == user_id,
                TicketDB.ticket_number == parsed["ticket_number"],
            )
            .first()
        )
        if existing_ticket:
            raise HTTPException(status_code=409, detail="This ticket was already added.")

        # Step 4: Create and save ticket in the database.
        new_ticket = TicketDB(
            user_id=user_id,
            city=parsed["city"],
            plate_number=parsed["plate_number"],
            ticket_number=parsed["ticket_number"],
            issued_at=issued_at_dt,
            payable_at=issued_at_dt + timedelta(hours=48),
            status="pending",
            fine_amount=parsed["fine_amount"],
        )
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)

        # Step 5: Return the created ticket.
        return to_ticket_schema(new_ticket)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ticket scan failed: {exc}") from exc
