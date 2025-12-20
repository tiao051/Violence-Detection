import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/providers/camera_provider.dart';
import 'package:security_app/theme/app_theme.dart';
import 'package:security_app/widgets/error_widget.dart' as error_widget;
import 'package:security_app/widgets/empty_state_widget.dart';
import 'package:security_app/widgets/camera_grid_item.dart';

/// Tab that displays the list of available cameras.
class CameraTab extends StatefulWidget {
  const CameraTab({super.key});

  @override
  State<CameraTab> createState() => _CameraTabState();
}

class _CameraTabState extends State<CameraTab> {
  late TextEditingController _searchController;

  @override
  void initState() {
    super.initState();
    _searchController = TextEditingController();
    // Defer fetch until after first frame to avoid calling provider during build.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final authProvider = context.read<AuthProvider>();
      final accessToken = authProvider.accessToken;

      if (accessToken == null || accessToken.isEmpty) {
        context.read<CameraProvider>().setErrorMessage('Not authenticated');
        return;
      }

      context.read<CameraProvider>().fetchCameras(accessToken: accessToken);
    });
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<CameraProvider>(
      builder: (context, cameraProvider, child) {
        if (cameraProvider.isLoading) {
          return const Center(
            child: SpinKitFadingCircle(
              color: kAccentColor,
              size: 50.0,
            ),
          );
        }

        if (cameraProvider.errorMessage != null) {
          return error_widget.ErrorWidget(
            errorMessage: cameraProvider.errorMessage ?? "Unknown error",
            onRetry: () {
              final authProvider = context.read<AuthProvider>();
              final accessToken = authProvider.accessToken;

              if (accessToken == null || accessToken.isEmpty) {
                return;
              }

              context.read<CameraProvider>().clearCache();
              context
                  .read<CameraProvider>()
                  .fetchCameras(accessToken: accessToken);
            },
            iconData: Icons.videocam_off,
            title: "Failed to Load Cameras",
          );
        }

        final cameras = cameraProvider.cameras;

        if (cameras.isEmpty) {
          return EmptyStateWidget(
            title: "No Cameras Available",
            subtitle: "Add cameras in settings to start monitoring",
            iconData: Icons.videocam_off,
          );
        }

        final filteredCameras = cameraProvider.filteredCameras;

        return Column(
          children: [
            // Modern Search Bar
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
              child: Container(
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: Colors.white.withOpacity(0.1)),
                ),
                child: TextField(
                  controller: _searchController,
                  onChanged: (query) {
                    cameraProvider.setSearchQuery(query);
                  },
                  style: const TextStyle(color: Colors.white),
                  decoration: InputDecoration(
                    hintText: 'Search cameras...',
                    hintStyle: TextStyle(color: Colors.white.withOpacity(0.5)),
                    prefixIcon: Icon(Icons.search,
                        color: Colors.white.withOpacity(0.7)),
                    suffixIcon: cameraProvider.searchQuery.isNotEmpty
                        ? IconButton(
                            icon:
                                const Icon(Icons.clear, color: Colors.white70),
                            onPressed: () {
                              _searchController.clear();
                              cameraProvider.clearSearch();
                            },
                          )
                        : null,
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 14,
                    ),
                  ),
                ),
              ),
            ),

            // Section Title
            if (filteredCameras.isNotEmpty)
              Padding(
                padding:
                    const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
                child: Row(
                  children: [
                    Text(
                      'Live Feeds',
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                            color: Colors.white.withOpacity(0.9),
                            fontWeight: FontWeight.bold,
                            letterSpacing: 0.5,
                          ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 8, vertical: 2),
                      decoration: BoxDecoration(
                        color: Theme.of(context)
                            .colorScheme
                            .primary
                            .withOpacity(0.2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        '${filteredCameras.length}',
                        style: TextStyle(
                          color: Theme.of(context).colorScheme.primary,
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

            // Camera list
            Expanded(
              child: RefreshIndicator(
                onRefresh: () {
                  final authProvider = context.read<AuthProvider>();
                  final accessToken = authProvider.accessToken;

                  if (accessToken == null || accessToken.isEmpty) {
                    return Future.error('Not authenticated');
                  }

                  return context
                      .read<CameraProvider>()
                      .refreshCameras(accessToken: accessToken);
                },
                color: Theme.of(context).colorScheme.primary,
                strokeWidth: 3.0,
                backgroundColor: Colors.transparent,
                child: filteredCameras.isEmpty
                    ? Center(
                        child: Text(
                          'No cameras match "${cameraProvider.searchQuery}"',
                          style: Theme.of(context).textTheme.bodyMedium,
                        ),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 16.0, vertical: 8.0),
                        itemCount: filteredCameras.length,
                        itemBuilder: (context, index) {
                          final camera = filteredCameras[index];
                          return Padding(
                            padding: const EdgeInsets.only(bottom: 16.0),
                            child: AspectRatio(
                              aspectRatio: 16 / 10,
                              child: CameraGridItem(
                                camera: camera,
                                onTap: () {
                                  context.push('/live_view/${camera.id}');
                                },
                              ),
                            ),
                          );
                        },
                      ),
              ),
            ),
          ],
        );
      },
    );
  }
}
