# Auth Feature Implementation Plan

## Overview

Complete authentication system with Sign Up, Login (Email/Password + Google), Change Password, and Edit Profile.

---

### **Phase 1: Core Models & Services**

**Step 1: Create AuthModel**
*   **File**: `lib/models/auth_model.dart`
*   **What**: Data class for user profile
*   **Status**: `NOT STARTED`
    *   Create file with fields: `uid`, `email`, `displayName`, `photoUrl`, `authProvider`, `createdAt`
    *   Add `toJson`/`fromJson` methods
    *   Add `copyWith` method
    *   Test serialization
*   **Commit**: `feat: add AuthModel for unified user profile data`

**Step 2: Update AuthService**
*   **File**: `lib/services/auth_service.dart`
*   **What**: Add 5 new methods
*   **Status**: `NOT STARTED`
    *   Add `signUpWithEmail(email, password, displayName)` → Firebase + Firestore
    *   Add `loginWithEmail(email, password)` → Firebase
    *   Add `changePassword(currentPassword, newPassword)` → Firebase (re-auth)
    *   Add `updateDisplayName(newName)` → Firebase + Firestore
    *   Add `getUserProfile()` → Fetch from Firestore
    *   Test each method, error handling
*   **Commit**: `feat: add auth service methods (sign up, change password, update profile)`

**Step 3: Update AuthProvider**
*   **File**: `lib/providers/auth_provider.dart`
*   **What**: Add 4 new methods to state management
*   **Status**: `NOT STARTED`
    *   Add state: `_user (AuthModel?)`, `_isLoadingSignUp`
    *   Add `signUp(email, password, displayName)` method
    *   Add `login(email, password)` method
    *   Add `changePassword(oldPwd, newPwd)` method
    *   Add `updateProfile(displayName)` method
    *   Test state changes, error handling
*   **Commit**: `feat: add auth provider methods (sign up, change password, update profile)`

---

### **Phase 2: UI Screens**

**Step 4: Create SignUpScreen**
*   **File**: `lib/screens/sign_up_screen.dart`
*   **What**: Registration screen
*   **Status**: `NOT STARTED`
    *   Create StatefulWidget with form
    *   4 TextFormFields: email, password, confirm password, display name
    *   Form validation (email format, password min 6 chars, match)
    *   Show/hide password toggle
    *   Loading state + error display
    *   "Already have account?" → `/login` button
    *   On success → redirect `/home`
*   **Test**:
    *   Invalid email → error
    *   Weak password → error
    *   Password mismatch → error
    *   Valid sign up → success + redirect
    *   Email exists → error
*   **Commit**: `feat: add SignUpScreen with email/password registration`

**Step 5: Update LoginScreen**
*   **File**: `lib/screens/login_screen.dart`
*   **What**: Add sign up navigation
*   **Status**: `PARTIALLY DONE`
    *   Update "Don't have account?" button → `/sign-up`
    *   Add "Forgot Password?" link → `/forgot-password`
    *   Test both buttons work
*   **Commit**: `fix: add sign up and forgot password navigation to LoginScreen`

**Step 6: Create ChangePasswordDialog**
*   **File**: `lib/dialogs/change_password_dialog.dart`
*   **What**: Dialog for password change
*   **Status**: `NOT STARTED`
    *   AlertDialog with 3 fields (current, new, confirm new)
    *   Validation + show/hide toggle
    *   Loading + error handling
*   **Test**:
    *   Wrong current password → error
    *   Mismatch → error
    *   Success → can re-login with new password
*   **Commit**: `feat: add ChangePasswordDialog for password management`

**Step 7: Update ProfileScreen**
*   **File**: `lib/screens/profile_screen.dart`
*   **What**: Add edit name + change password
*   **Status**: `PARTIALLY DONE (logout works)`
    *   Inline edit for display name (tap to edit, save/cancel)
    *   "Change Password" → ChangePasswordDialog
*   **Test**:
    *   Edit name → saves → reflects on profile
    *   Change password → success → can re-login
    *   Logout works
*   **Commit**: `feat: add edit display name and change password to ProfileScreen`

---

### **Phase 3: Firestore Schema**

**Step 8: Setup Firestore Collections**
*   **Collections**: `users/{uid}`
*   **Status**: `NOT STARTED`
*   Create document structure in Firebase Console:
    ```json
    {
      "uid": "string",
      "email": "string",
      "displayName": "string",
      "photoUrl": "string|null",
      "authProvider": "email|google",
      "createdAt": "timestamp",
      "updatedAt": "timestamp"
    }
    ```
*   Set Security Rules
    *   **Firestore Security Rules**:
        ```
        match /users/{uid} {
          allow read, write: if request.auth.uid == uid;
        }
        ```
---

### **Phase 4: Routing**

**Step 9: Update GoRouter**
*   **File**: `lib/main.dart`
*   **What**: Add routes
*   **Status**: `PARTIALLY DONE`
    *   Add `/sign-up` → `SignUpScreen`
    *   Add `/forgot-password` → `ForgotPasswordScreen`
    *   Test routes work
*   **Commit**: `feat: add sign up and forgot password routes to GoRouter`

---

### **Phase 5: Optional**

**Step 10 (Optional): Create ForgotPasswordScreen**
*   **File**: `lib/screens/forgot_password_screen.dart`
*   **What**: Password reset via email
*   **Status**: `NOT STARTED`
    *   Email input field
    *   "Send Reset Link" button
    *   Call `FirebaseAuth.sendPasswordResetEmail()`
    *   Show success message
    *   Test email reset works
*   **Commit**: `feat: add ForgotPasswordScreen for password reset`

---

### **Implementation Order**

1.  **Step 1**: AuthModel → Test → Commit
2.  **Step 2**: AuthService → Test → Commit
3.  **Step 3**: AuthProvider → Test → Commit
4.  **Step 4**: SignUpScreen → Test → Commit
5.  **Step 5**: LoginScreen → Test → Commit
6.  **Step 6**: ChangePasswordDialog → Test → Commit
7.  **Step 7**: ProfileScreen → Test → Commit
8.  **Step 8**: Firestore Setup (manual)
9.  **Step 9**: GoRouter → Test → Commit
10. **Step 10 (Optional)**: ForgotPasswordScreen → Test → Commit

---

### **Testing Checklist**

*   **SignUp**
    *   [ ] Valid data → Success
    *   [ ] Email exists → Error
    *   [ ] Weak password → Error
    *   [ ] Password mismatch → Error
*   **Login**
    *   [ ] Correct credentials → Success
    *   [ ] Wrong password → Error
    *   [ ] Google Sign-In → Success
*   **Change Password**
    *   [ ] Correct current pwd + matching new → Success
    *   [ ] Wrong current pwd → Error
    *   [ ] After change: re-login with new pwd → Success
*   **Edit Display Name**
    *   [ ] Edit → Save → Reflects immediately
    *   [ ] Edit → Cancel → No change

---

### **Current Status**

*   **Done**: Profile Screen, Google Sign-In
*   **TODO**: Steps 1-10