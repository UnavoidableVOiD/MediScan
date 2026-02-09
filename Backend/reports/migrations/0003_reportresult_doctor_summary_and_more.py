from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reports', '0002_reportresult'),
    ]

    operations = [
        migrations.AddField(
            model_name='reportresult',
            name='doctor_summary',
            field=models.TextField(blank=True, help_text='AI generated summary for the doctor', null=True),
        ),
        migrations.AlterField(
            model_name='reportresult',
            name='summary',
            field=models.TextField(help_text='AI generated summary for the patient'),
        ),
    ]
