# crud.py
from tuneparam.database.schema.models import Student

def create_student(session, name, age):
    student = Student(name=name, age=age)
    session.add(student)
    session.commit()

def read_students(session):
    students = session.query(Student).all()
    return students

def update_student(session, student_id, new_name=None, new_age=None):
    student = session.query(Student).get(student_id)
    if student:
        if new_name:
            student.name = new_name
        if new_age:
            student.age = new_age
        session.commit()

def delete_student(session, student_id):
    student = session.query(Student).get(student_id)
    if student:
        session.delete(student)
        session.commit()
