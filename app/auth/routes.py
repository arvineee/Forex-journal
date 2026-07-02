from flask import render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, current_user, login_required
from app import db
from app.auth import auth_bp
from app.models import User
from app.auth.forms import LoginForm, RegistrationForm, RequestResetForm, ResetPasswordForm
from app.email import send_reset_email

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now registered!', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html', title='Register', form=form)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        return redirect(next_page or url_for('dashboard.index'))
    
    return render_template('auth/login.html', title='Sign In', form=form)

@auth_bp.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('main.index'))


@auth_bp.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    """Step 1: user submits their email to receive a reset link."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))

    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_reset_email(user)
        # Same message whether or not the email exists, to avoid leaking
        # which emails are registered. (validate_email above already checks
        # existence for UX purposes; this line is just defense in depth.)
        flash('If that email is registered, a reset link has been sent to it.', 'info')
        return redirect(url_for('auth.login'))

    return render_template('auth/reset_request.html', title='Reset Password', form=form)


@auth_bp.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    """Step 2: user follows the emailed link and sets a new password."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))

    user = User.verify_reset_token(token)
    if user is None:
        flash('That reset link is invalid or has expired.', 'warning')
        return redirect(url_for('auth.reset_request'))

    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        db.session.commit()
        flash('Your password has been updated! You can now log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/reset_password.html', title='Reset Password', form=form)

