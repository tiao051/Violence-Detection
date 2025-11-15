import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:go_router/go_router.dart';
import 'package:package_info_plus/package_info_plus.dart';
import 'package:security_app/providers/profile_provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/theme/app_theme.dart';

/// Screen displaying user profile and account settings
class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  String _appVersion = '';

  @override
  void initState() {
    super.initState();
    _loadAppVersion();
    // Load profile when screen opens
    Future.microtask(() {
      context.read<ProfileProvider>().loadProfile();
    });
  }

  Future<void> _loadAppVersion() async {
    final packageInfo = await PackageInfo.fromPlatform();
    setState(() {
      _appVersion = packageInfo.version;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile'),
        centerTitle: true,
      ),
      body: Consumer<ProfileProvider>(
        builder: (context, profileProvider, _) {
          if (profileProvider.isLoading) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }

          if (profileProvider.errorMessage != null) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.error_outline, size: 48, color: Colors.red),
                  const SizedBox(height: 16),
                  Text(
                    'Error: ${profileProvider.errorMessage}',
                    textAlign: TextAlign.center,
                    style: const TextStyle(color: Colors.red),
                  ),
                  const SizedBox(height: 24),
                  ElevatedButton(
                    onPressed: () {
                      profileProvider.refreshProfile();
                    },
                    child: const Text('Retry'),
                  ),
                ],
              ),
            );
          }

          final profile = profileProvider.profile;

          if (profile == null) {
            return const Center(
              child: Text('No profile data'),
            );
          }

          return SingleChildScrollView(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                // ===== User Info Card =====
                _buildUserInfoCard(profile, context),
                const SizedBox(height: 24),

                // ===== Account Section =====
                _buildAccountSection(context),
                const SizedBox(height: 24),

                // ===== About Section =====
                _buildAboutSection(),
                const SizedBox(height: 24),

                // ===== Actions Section =====
                _buildActionsSection(context),
                const SizedBox(height: 24),
              ],
            ),
          );
        },
      ),
    );
  }

  /// User Info Card - Display profile picture, name, and email
  Widget _buildUserInfoCard(dynamic profile, BuildContext context) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(24.0),
        child: Column(
          children: [
            // Profile picture
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: kAppGradient,
              ),
              child: profile.photoUrl != null
                  ? CircleAvatar(
                      backgroundImage: NetworkImage(profile.photoUrl!),
                    )
                  : CircleAvatar(
                      backgroundColor: Theme.of(context).colorScheme.primary,
                      child: const Icon(Icons.person, size: 50, color: Colors.white),
                    ),
            ),
            const SizedBox(height: 16),

            // Display name
            Text(
              profile.displayName ?? 'User',
              style: Theme.of(context).textTheme.headlineSmall,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),

            // Email
            Text(
              profile.email,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey,
                  ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  /// Account Section - Edit profile options
  Widget _buildAccountSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Account',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 12),
        ListTile(
          leading: const Icon(Icons.edit),
          title: const Text('Edit Profile'),
          subtitle: const Text('Update your information'),
          trailing: const Icon(Icons.arrow_forward_ios, size: 16),
          onTap: () {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Edit profile feature coming soon')),
            );
          },
        ),
        ListTile(
          leading: const Icon(Icons.security),
          title: const Text('Change Password'),
          subtitle: const Text('Update your password'),
          trailing: const Icon(Icons.arrow_forward_ios, size: 16),
          onTap: () {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Change password feature coming soon')),
            );
          },
        ),
      ],
    );
  }

  /// About Section - App info
  Widget _buildAboutSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'About',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 12),
        ListTile(
          leading: const Icon(Icons.info_outline),
          title: const Text('App Version'),
          trailing: Text(
            _appVersion,
            style: const TextStyle(color: Colors.grey),
          ),
        ),
        ListTile(
          leading: const Icon(Icons.description),
          title: const Text('Terms of Service'),
          trailing: const Icon(Icons.arrow_forward_ios, size: 16),
          onTap: () {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Terms of Service feature coming soon')),
            );
          },
        ),
        ListTile(
          leading: const Icon(Icons.privacy_tip),
          title: const Text('Privacy Policy'),
          trailing: const Icon(Icons.arrow_forward_ios, size: 16),
          onTap: () {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(content: Text('Privacy Policy feature coming soon')),
            );
          },
        ),
      ],
    );
  }

  /// Actions Section - Logout
  Widget _buildActionsSection(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Actions',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 12),
            ),
            icon: const Icon(Icons.logout),
            label: const Text('Logout'),
            onPressed: () {
              _showLogoutConfirmDialog(context);
            },
          ),
        ),
      ],
    );
  }

  /// Confirm dialog for logout
  void _showLogoutConfirmDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Confirm Logout'),
        content: const Text('Are you sure you want to logout?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(context);
              
              // Logout - AuthProvider handles clearing auth state
              await context.read<AuthProvider>().logout();

              // Navigate to login screen
              if (mounted) {
                context.go('/login');
              }
            },
            child: const Text('Logout', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}
