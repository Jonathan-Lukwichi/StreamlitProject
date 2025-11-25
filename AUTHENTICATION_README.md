# HealthForecast AI - Authentication System

## ⚠️ PROTOTYPE DEMONSTRATION ONLY

This authentication system is designed for **prototype demonstration purposes** to showcase how the application would work in a real-world scenario with role-based access control.

**IMPORTANT:** This is a **UI-level authentication** system and provides **NO real security**. The authentication checks happen client-side in Streamlit and can be bypassed by anyone with technical knowledge.

For production deployment with real healthcare data, you **MUST** implement proper server-side authentication using:
- **Supabase Auth** with Row Level Security (RLS)
- **Streamlit Enterprise** authentication
- **OAuth2/OIDC** with an auth proxy (e.g., nginx)

---

## Demo Credentials

### Administrator Access
- **Username:** `admin`
- **Password:** `admin123`
- **Access:** All pages (02-13 including Data Hub, EDA, Modeling, Results, etc.)

### User Access
- **Username:** `user1` or `user2`
- **Password:** `user123`
- **Access:** Dashboard (01) + Forecast pages (10-13) only

---

## How It Works

### 1. **Welcome Page (Login)**
- Initial landing page with authentication form
- Sidebar is **hidden** until login
- Displays prominent "PROTOTYPE" warning
- Shows demo credentials for easy testing
- Redirects to appropriate page after successful login

### 2. **Role-Based Page Access**

#### Admin Pages (02-09)
- 02 - Data Hub
- 03 - Data Preparation Studio
- 04 - EDA (Exploratory Data Analysis)
- 05 - Benchmarks
- 06 - Advanced Feature Engineering
- 07 - Automated Feature Selection
- 08 - Modeling Hub
- 09 - Results

**Protection:** These pages check for both authentication AND admin role. Regular users attempting to access these pages will see an error message and be redirected.

#### User + Admin Pages (01, 10-13)
- 01 - Dashboard
- 10 - Forecast
- 11 - Staff Scheduling Optimization
- 12 - Inventory Management Optimization
- 13 - Decision Command Center

**Protection:** These pages only require authentication (any logged-in user can access them).

### 3. **Sidebar Navigation**
The sidebar shows all pages for simplicity in this prototype. Page-level authentication checks handle access control:
- **Unauthenticated users:** Attempting to access any page will redirect to login
- **User role:** Can access pages 01, 10-13 (others show "Admin Required" error)
- **Admin role:** Can access all pages (01-13)

**Note:** In a production system, you would implement dynamic sidebar hiding to improve UX, but for this prototype, showing all pages with page-level protection is acceptable.

### 4. **Logout Functionality**
- Logout button appears in sidebar for authenticated users
- Shows current user name and role
- Clears all session state on logout
- Redirects to Welcome page

---

## Technical Architecture

### File Structure
```
app_core/
└── auth/
    ├── __init__.py                # Package exports
    ├── authentication.py          # Core authentication logic
    └── navigation.py              # Sidebar navigation management

Welcome.py                         # Login page
pages/
├── 01_Dashboard.py               # User + Admin
├── 02_Data_Hub.py                # Admin only
├── 03_Data_Preparation_Studio.py # Admin only
├── 04_EDA.py                     # Admin only
├── 05_Benchmarks.py              # Admin only
├── 06_Advanced_Feature_Engineering.py  # Admin only
├── 07_Automated_Feature_Selection.py   # Admin only
├── 08_Modeling_Hub.py            # Admin only
├── 09_Results.py                 # Admin only
├── 10_Forecast.py                # User + Admin
├── 11_Staff_Scheduling_Optimization.py # User + Admin
├── 12_Inventory_Management_Optimization.py # User + Admin
└── 13_Decision_Command_Center.py # User + Admin
```

### Key Components

#### 1. **authentication.py**
- User credential management (bcrypt password hashing)
- Authentication state checking
- Role-based access control functions
- Page protection decorators:
  - `require_authentication()` - For user + admin pages
  - `require_admin_access()` - For admin-only pages

#### 2. **navigation.py**
- Logout button with user info display in sidebar
- Shows current user name and role
- Session state clearing on logout
- Simplified approach (all pages visible in sidebar, protection at page level)

#### 3. **Session State Management**
Authenticated users have these session variables:
- `authenticated` - Boolean flag
- `username` - User's login name
- `name` - User's display name
- `role` - User role ("admin" or "user")
- `email` - User's email address

---

## Usage in Pages

### Admin-Only Pages (02-09)
```python
from app_core.auth.authentication import require_admin_access
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button

require_admin_access()  # Blocks non-admin users
configure_sidebar_navigation()  # Sets up sidebar (currently a pass-through)

# In your page body function:
def page_data_hub():
    # ... apply theme and sidebar
    add_logout_button()  # Shows user info and logout in sidebar
    # ... rest of page content
```

### User + Admin Pages (01, 10-13)
```python
from app_core.auth.authentication import require_authentication
from app_core.auth.navigation import configure_sidebar_navigation, add_logout_button

require_authentication()  # Requires any authenticated user
configure_sidebar_navigation()  # Sets up sidebar (currently a pass-through)

# In your page body function:
def page_dashboard():
    # ... apply theme and sidebar
    add_logout_button()  # Shows user info and logout in sidebar
    # ... rest of page content
```

---

## Adding New Users

To add new users, edit `app_core/auth/authentication.py`:

1. **Generate Password Hash:**
```python
from app_core.auth.authentication import hash_password
print(hash_password("your_password_here"))
```

2. **Add to Credentials:**
```python
def get_user_credentials():
    credentials = {
        "usernames": {
            "newuser": {
                "name": "New User Name",
                "password": "$2b$12$...",  # Generated hash
                "role": "user",  # or "admin"
                "email": "newuser@example.com"
            }
        }
    }
    return credentials
```

---

## Security Considerations

### Current Limitations (Prototype)
❌ Client-side authentication only
❌ Session state can be manipulated
❌ No secure token management
❌ No audit logging
❌ Passwords stored in code
❌ No session timeout enforcement
❌ No account lockout
❌ No multi-factor authentication

### Production Requirements
✅ Server-side authentication
✅ Encrypted connections (HTTPS)
✅ Secure credential storage
✅ Session management with timeouts
✅ Audit logging for HIPAA compliance
✅ Role-based Row Level Security (RLS)
✅ Multi-factor authentication (MFA)
✅ Account management (password reset, etc.)
✅ Penetration testing
✅ Regular security audits

---

## Migration to Production

When moving to production, replace this system with:

### Option 1: Supabase Auth + RLS
```python
# Example Supabase Auth integration
from supabase import create_client
supabase = create_client(url, key)

# Login
user = supabase.auth.sign_in_with_password({
    "email": email,
    "password": password
})

# Queries automatically enforce RLS based on user role
data = supabase.table("patients").select("*").execute()
```

### Option 2: Streamlit Enterprise
- Built-in authentication
- SSO/SAML support
- Workspace management
- Audit logs
- Contact: https://streamlit.io/enterprise

### Option 3: Auth Proxy
- nginx + OAuth2 Proxy
- Keycloak
- Auth0
- Okta

---

## Testing the Authentication

### Test Scenario 1: Admin Flow
1. Go to Welcome page
2. Login as `admin` / `admin123`
3. Verify you see all pages (01-13) in sidebar
4. Navigate to Data Hub (02) - should work
5. Navigate to Modeling Hub (08) - should work
6. Click Logout

### Test Scenario 2: User Flow
1. Go to Welcome page
2. Login as `user1` / `user123`
3. Verify you only see pages 01, 10, 11, 12, 13 in sidebar
4. Navigate to Dashboard (01) - should work
5. Navigate to Forecast (10) - should work
6. Try to manually navigate to `/pages/02_Data_Hub.py` - should be blocked
7. Click Logout

### Test Scenario 3: Unauthenticated Access
1. Without logging in, try to access any page directly
2. Should be redirected to Welcome page with login prompt

---

## Libraries Used

- **streamlit-authenticator** - Authentication widgets and password hashing
- **bcrypt** - Password hashing (via streamlit-authenticator)

Install with:
```bash
pip install streamlit-authenticator
```

**Note:** st-pages was originally planned for dynamic sidebar management but is not used in the current simplified implementation. Page-level protection handles access control instead.

---

## Support

For questions about this prototype authentication system:
1. Review this README
2. Check the code comments in `app_core/auth/`
3. Test with the demo credentials provided

Remember: This is for **demonstration only**. Implement proper security before handling real healthcare data!

---

**Last Updated:** 2025-11-25
**Status:** ✅ Prototype Implementation Complete
