import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import '../../providers/camera_provider.dart';
import '../../widgets/error_widget.dart' as error_widget;
import '../../widgets/empty_state_widget.dart';

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
    // Use read() instead of watch() because we don't need to rebuild on changes.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<CameraProvider>().fetchCameras();
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
              cameraProvider.clearCache();
              cameraProvider.fetchCameras();
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
                onRefresh: () => cameraProvider.refreshCameras(),
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
                        itemCount: filteredCameras.length,
                        itemBuilder: (context, index) {
                          final camera = filteredCameras[index];

                          return Card(
                            margin: const EdgeInsets.all(8.0),
                            child: ListTile(
                              leading: const CircleAvatar(
                                child: Icon(Icons.videocam_outlined),
                              ),
                              title: Text(camera.name),
                              subtitle: Text('ID: ${camera.id}'),
                              trailing: const Icon(Icons.chevron_right),
                              onTap: () {
                                context.push('/live_view/${camera.id}');
                              },
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