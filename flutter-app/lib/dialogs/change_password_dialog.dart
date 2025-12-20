import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/theme/app_theme.dart';

/// A dialog widget for changing the user's password.
///
/// This dialog provides fields for current password, new password,
/// and confirming the new password. It uses [AuthProvider] to handle
/// the password change logic and displays loading/error states.
///
/// To show this dialog, use:
/// ```
/// showDialog(
///   context: context,
///   builder: (context) => const ChangePasswordDialog(),
/// );
/// ```
class ChangePasswordDialog extends StatefulWidget {
  const ChangePasswordDialog({super.key});

  @override
  State<ChangePasswordDialog> createState() => _ChangePasswordDialogState();
}

class _ChangePasswordDialogState extends State<ChangePasswordDialog> {
  final _formKey = GlobalKey<FormState>();
  late TextEditingController _currentPasswordController;
  late TextEditingController _newPasswordController;
  late TextEditingController _confirmPasswordController;

  bool _obscureCurrent = true;
  bool _obscureNew = true;
  bool _obscureConfirm = true;

  @override
  void initState() {
    super.initState();
    _currentPasswordController = TextEditingController();
    _newPasswordController = TextEditingController();
    _confirmPasswordController = TextEditingController();
  }

  @override
  void dispose() {
    _currentPasswordController.dispose();
    _newPasswordController.dispose();
    _confirmPasswordController.dispose();
    super.dispose();
  }

  /// New password validator
  String? _validateNewPassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'New password is required';
    }
    if (value.length < 6) {
      return 'Password must be at least 6 characters';
    }
    return null;
  }

  /// Confirm password validator
  String? _validateConfirmPassword(String? value) {
    if (value == null || value.isEmpty) {
      return 'Please confirm your new password';
    }
    if (value != _newPasswordController.text) {
      return 'Passwords do not match';
    }
    return null;
  }

  /// Handles the password change submission
  Future<void> _handleChangePassword() async {
    // 1. Validate the form
    if (!_formKey.currentState!.validate()) {
      return;
    }

    final authProvider = context.read<AuthProvider>();

    try {
      // 2. Call the provider
      await authProvider.changePassword(
        currentPassword: _currentPasswordController.text,
        newPassword: _newPasswordController.text,
      );

      // 3. Handle success
      if (mounted) {
        Navigator.pop(context); // Close the dialog
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Password changed successfully!'),
            backgroundColor: kSuccessColor,
          ),
        );
      }
    } catch (e) {
      // 4. Handle error - Clean up error message
      if (mounted) {
        // Extract clean error message
        String errorMessage = e.toString();
        // Remove "Exception: " prefix if present
        if (errorMessage.startsWith('Exception: ')) {
          errorMessage = errorMessage.substring('Exception: '.length);
        }

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(errorMessage),
            backgroundColor: kErrorColor,
            duration: const Duration(seconds: 4),
          ),
        );

        // Clear error message after a short delay to prevent redirect issues
        Future.delayed(const Duration(seconds: 5), () {
          if (mounted) {
            authProvider.clearError();
          }
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    // Use Consumer to react to loading state changes
    return Consumer<AuthProvider>(
      builder: (context, auth, child) {
        return AlertDialog(
          title: const Text('Change Password'),
          content: Form(
            key: _formKey,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // 1. Current Password
                TextFormField(
                  controller: _currentPasswordController,
                  decoration: InputDecoration(
                    labelText: 'Current Password',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.lock_outline),
                    suffixIcon: IconButton(
                      icon: Icon(_obscureCurrent
                          ? Icons.visibility_off
                          : Icons.visibility),
                      onPressed: () =>
                          setState(() => _obscureCurrent = !_obscureCurrent),
                    ),
                  ),
                  obscureText: _obscureCurrent,
                  validator: (val) => val == null || val.isEmpty
                      ? 'Current password is required'
                      : null,
                  enabled: !auth.isLoading, // Disable when loading
                ),
                const SizedBox(height: 16),

                // 2. New Password
                TextFormField(
                  controller: _newPasswordController,
                  decoration: InputDecoration(
                    labelText: 'New Password',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.lock),
                    suffixIcon: IconButton(
                      icon: Icon(_obscureNew
                          ? Icons.visibility_off
                          : Icons.visibility),
                      onPressed: () =>
                          setState(() => _obscureNew = !_obscureNew),
                    ),
                  ),
                  obscureText: _obscureNew,
                  validator: _validateNewPassword,
                  enabled: !auth.isLoading, // Disable when loading
                ),
                const SizedBox(height: 16),

                // 3. Confirm New Password
                TextFormField(
                  controller: _confirmPasswordController,
                  decoration: InputDecoration(
                    labelText: 'Confirm New Password',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.lock),
                    suffixIcon: IconButton(
                      icon: Icon(_obscureConfirm
                          ? Icons.visibility_off
                          : Icons.visibility),
                      onPressed: () =>
                          setState(() => _obscureConfirm = !_obscureConfirm),
                    ),
                  ),
                  obscureText: _obscureConfirm,
                  validator: _validateConfirmPassword,
                  enabled: !auth.isLoading, // Disable when loading
                ),
              ],
            ),
          ),
          actions: [
            // Cancel Button
            TextButton(
              onPressed: auth.isLoading ? null : () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),

            // Save Button (Themed)
            if (auth.isLoading)
              const Padding(
                padding: EdgeInsets.all(12.0),
                child: SpinKitFadingCircle(
                  color: kAccentColor,
                  size: 24.0,
                ),
              )
            else
              ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: kSuccessColor,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(AppRadius.md),
                  ),
                ),
                onPressed: _handleChangePassword,
                child: const Text('Save'),
              ),
          ],
        );
      },
    );
  }
}
