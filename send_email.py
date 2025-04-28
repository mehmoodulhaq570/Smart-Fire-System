# Firstly you have to get the app password from your google account
# replace the password withe your own password and then you can use it in the code below

import smtplib
from email.mime.text import MIMEText

def send_email_alert():
    sender = "mehmoodulhaq1040@gmail.com"
    receiver = "kashifmuneer1085@gmail.com"
    password = "shauoyxkhpbcqlyq"

    subject = "FIRE ALERT"
    body = "Fire has been detected. Please take immediate action!"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, msg.as_string())
    
    print("Email sent!")

send_email_alert()