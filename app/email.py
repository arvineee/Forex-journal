"""
Save this file as app/email.py (top-level, next to models.py — NOT inside app/auth/)
"""
from flask import current_app, render_template, url_for
from flask_mail import Message
from app import mail


def send_reset_email(user):
    """Send a password reset email containing a signed, expiring token link."""
    token = user.get_reset_token()
    reset_url = url_for('auth.reset_token', token=token, _external=True)

    msg = Message(
        subject='Password Reset Request - Forex Journal',
        sender=current_app.config['MAIL_DEFAULT_SENDER'],
        recipients=[user.email]
    )
    msg.body = (
        f"Hi {user.username},\n\n"
        "You (or someone else) requested a password reset for your Forex Journal account.\n\n"
        f"Click the link below to reset your password. This link expires in 30 minutes:\n"
        f"{reset_url}\n\n"
        "If you did not make this request, simply ignore this email and no changes will be made.\n"
    )
    # Optional HTML version - only kept if templates/email/reset_password.html exists
    try:
        msg.html = render_template('email/reset_password.html', user=user, reset_url=reset_url)
    except Exception:
        pass

    mail.send(msg)

