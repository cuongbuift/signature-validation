from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from database import get_db
from models import Employee
from schemas import EmployeeCreate, EmployeeUpdate, EmployeeOut

router = APIRouter(prefix="/employees", tags=["Employees"])


@router.post("", response_model=EmployeeOut, status_code=status.HTTP_201_CREATED)
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)):
    if db.query(Employee).filter(Employee.employee_code == payload.employee_code).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Mã nhân viên '{payload.employee_code}' đã tồn tại.",
        )
    emp = Employee(employee_code=payload.employee_code, full_name=payload.full_name)
    db.add(emp)
    db.commit()
    db.refresh(emp)
    return _to_out(emp)


@router.get("", response_model=list[EmployeeOut])
def list_employees(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    employees = db.query(Employee).filter(Employee.is_active == True).offset(skip).limit(limit).all()
    return [_to_out(e) for e in employees]


@router.get("/{employee_code}", response_model=EmployeeOut)
def get_employee(employee_code: str, db: Session = Depends(get_db)):
    emp = _get_or_404(employee_code, db)
    return _to_out(emp)


@router.put("/{employee_code}", response_model=EmployeeOut)
def update_employee(employee_code: str, payload: EmployeeUpdate, db: Session = Depends(get_db)):
    emp = _get_or_404(employee_code, db)
    if payload.full_name is not None:
        emp.full_name = payload.full_name
    db.commit()
    db.refresh(emp)
    return _to_out(emp)


@router.delete("/{employee_code}", status_code=status.HTTP_204_NO_CONTENT)
def deactivate_employee(employee_code: str, db: Session = Depends(get_db)):
    emp = _get_or_404(employee_code, db)
    emp.is_active = False
    db.commit()


# ── helpers ──────────────────────────────────────────────────────────────────

def _get_or_404(employee_code: str, db: Session) -> Employee:
    emp = db.query(Employee).filter(Employee.employee_code == employee_code).first()
    if not emp:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy nhân viên '{employee_code}'.")
    return emp


def _to_out(emp: Employee) -> EmployeeOut:
    return EmployeeOut(
        id=emp.id,
        employee_code=emp.employee_code,
        full_name=emp.full_name,
        created_at=emp.created_at,
        is_active=emp.is_active,
        signature_count=len(emp.signatures),
    )
