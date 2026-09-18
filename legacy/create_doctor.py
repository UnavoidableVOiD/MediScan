from authentication.models import CustomUser
if not CustomUser.objects.filter(email='doctor@example.com').exists():
    user = CustomUser.objects.create_user(
        username='doctor',
        email='doctor@example.com',
        password='password123',
        first_name='John',
        last_name='Doe',
        role='doctor',
        doctor_status='VERIFIED'
    )
    user.save()
    print("Doctor created")
else:
    print("Doctor exists")
