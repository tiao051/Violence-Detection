import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/auth_provider.dart';
import 'package:security_app/providers/camera_provider.dart';
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
          return Center(
            child: SpinKitFadingCircle(
              color: Theme.of(context).colorScheme.primary,
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
              context.read<CameraProvider>().fetchCameras(accessToken: accessToken);
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
            // Search bar
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: TextField(
                controller: _searchController,
                onChanged: (query) {
                  cameraProvider.setSearchQuery(query);
                },
                decoration: InputDecoration(
                  hintText: 'Search cameras...',
                  prefixIcon: const Icon(Icons.search),
                  suffixIcon: cameraProvider.searchQuery.isNotEmpty
                      ? IconButton(
                          icon: const Icon(Icons.clear),
                          onPressed: () {
                            _searchController.clear();
                            cameraProvider.clearSearch();
                          },
                        )
                      : null,
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 16,
                    vertical: 12,
                  ),
                ),
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
                  
                  return context.read<CameraProvider>().refreshCameras(accessToken: accessToken);
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
                    : GridView.builder(
                        padding: const EdgeInsets.all(8.0),
                        gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                          crossAxisCount: 2,
                          childAspectRatio: 16 / 9,
                          crossAxisSpacing: 8.0,
                          mainAxisSpacing: 8.0,
                        ),
                        itemCount: filteredCameras.length,
                        itemBuilder: (context, index) {
                          final camera = filteredCameras[index];
                          return CameraGridItem(
                            camera: camera,
                            onTap: () {
                              context.push('/live_view/${camera.id}');
                            },
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