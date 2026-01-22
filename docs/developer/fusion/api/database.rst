.. _api-database:

========
Database
========

This page documents the database layer in ``fusion/api/db/``.

----

database.py - SQLAlchemy Setup
==============================

:Location: ``fusion/api/db/database.py``

Configures SQLAlchemy for the FUSION GUI.

Components
----------

**Engine and Session**

.. code-block:: python

   from sqlalchemy import create_engine
   from sqlalchemy.orm import sessionmaker, DeclarativeBase

   # SQLite database file in the API directory
   DATABASE_URL = "sqlite:///./fusion/api/fusion.db"

   engine = create_engine(
       DATABASE_URL,
       connect_args={"check_same_thread": False}  # Required for SQLite
   )

   SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

**Base Class**

.. code-block:: python

   class Base(DeclarativeBase):
       """Base class for all ORM models."""
       pass

**Dependency Injection**

.. code-block:: python

   def get_db() -> Generator[Session, None, None]:
       """
       FastAPI dependency that provides a database session.

       Yields a session and ensures it's closed after the request.
       """
       db = SessionLocal()
       try:
           yield db
       finally:
           db.close()

   # Usage in routes
   @router.get("/runs")
   def list_runs(db: Session = Depends(get_db)):
       return db.query(Run).all()

**Database Initialization**

.. code-block:: python

   def init_db() -> None:
       """
       Initialize the database, creating tables if they don't exist.

       Called during application startup (lifespan context manager).
       """
       Base.metadata.create_all(bind=engine)

----

models.py - ORM Models
======================

:Location: ``fusion/api/db/models.py``

Defines the SQLAlchemy ORM models.

Run Model
---------

The ``Run`` model tracks simulation run metadata and progress.

.. code-block:: python

   class Run(Base):
       """SQLAlchemy model for simulation runs."""

       __tablename__ = "runs"

       # Primary key - 12 character hex string
       id: Mapped[str] = mapped_column(String(12), primary_key=True)

       # User-provided name (optional)
       name: Mapped[str | None] = mapped_column(String(255), nullable=True)

       # Status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
       status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

       # Configuration stored as JSON string
       config_json: Mapped[str] = mapped_column(Text, nullable=False)

       # Template used to create this run
       template: Mapped[str] = mapped_column(String(100), nullable=False)

       # Process tracking (for cancellation)
       pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
       pgid: Mapped[int | None] = mapped_column(Integer, nullable=True)

       # Timestamps
       created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
       started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
       completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

       # Error information (if FAILED)
       error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

       # Progress cache (updated from progress.jsonl)
       current_erlang: Mapped[float | None] = mapped_column(Float, nullable=True)
       total_erlangs: Mapped[int | None] = mapped_column(Integer, nullable=True)
       current_iteration: Mapped[int | None] = mapped_column(Integer, nullable=True)
       total_iterations: Mapped[int | None] = mapped_column(Integer, nullable=True)

Field Descriptions
------------------

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``id``
     - String(12)
     - Primary key, 12-character hex string (e.g., ``abc123def456``)
   * - ``name``
     - String(255)
     - Optional user-friendly display name
   * - ``status``
     - String(20)
     - Run status: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
   * - ``config_json``
     - Text
     - Full INI configuration serialized as JSON
   * - ``template``
     - String(100)
     - Name of the template used (e.g., ``minimal.ini``)
   * - ``pid``
     - Integer
     - Process ID of the simulation subprocess
   * - ``pgid``
     - Integer
     - Process group ID (used for cancellation)
   * - ``created_at``
     - DateTime
     - When the run was created
   * - ``started_at``
     - DateTime
     - When the simulation process started
   * - ``completed_at``
     - DateTime
     - When the simulation finished (success or failure)
   * - ``error_message``
     - Text
     - Error details if status is FAILED
   * - ``current_erlang``
     - Float
     - Current Erlang load being simulated
   * - ``total_erlangs``
     - Integer
     - Total number of Erlang values to simulate
   * - ``current_iteration``
     - Integer
     - Current iteration within the Erlang value
   * - ``total_iterations``
     - Integer
     - Total iterations per Erlang value

Status Values
-------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Status
     - Description
   * - ``PENDING``
     - Run created but simulation not yet started
   * - ``RUNNING``
     - Simulation process is active
   * - ``COMPLETED``
     - Simulation finished successfully
   * - ``FAILED``
     - Simulation encountered an error
   * - ``CANCELLED``
     - User cancelled the run

Schema Diagram
--------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │                          runs                               │
   ├─────────────────────────────────────────────────────────────┤
   │ id              VARCHAR(12)   PK                            │
   │ name            VARCHAR(255)  NULLABLE                      │
   │ status          VARCHAR(20)   NOT NULL, INDEXED             │
   │ config_json     TEXT          NOT NULL                      │
   │ template        VARCHAR(100)  NOT NULL                      │
   │ pid             INTEGER       NULLABLE                      │
   │ pgid            INTEGER       NULLABLE                      │
   │ created_at      DATETIME      NOT NULL                      │
   │ started_at      DATETIME      NULLABLE                      │
   │ completed_at    DATETIME      NULLABLE                      │
   │ error_message   TEXT          NULLABLE                      │
   │ current_erlang  FLOAT         NULLABLE                      │
   │ total_erlangs   INTEGER       NULLABLE                      │
   │ current_iteration INTEGER     NULLABLE                      │
   │ total_iterations INTEGER      NULLABLE                      │
   └─────────────────────────────────────────────────────────────┘

Database Location
-----------------

The SQLite database file is stored at:

.. code-block:: text

   fusion/api/fusion.db

This file is created automatically on first run and should be excluded from
version control (already in ``.gitignore``).

Migrations
----------

Currently, FUSION uses SQLAlchemy's ``create_all()`` for schema management.
Tables are created automatically if they don't exist.

For schema changes in development:

1. Delete ``fusion.db`` to reset
2. Restart the GUI to recreate tables

.. note::

   For production deployments, consider using Alembic for proper migrations.
   This would allow schema changes without data loss.
