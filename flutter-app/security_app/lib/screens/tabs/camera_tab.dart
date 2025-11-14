import 'package:flutter/material.dart';
import 'package:flutter_spinkit/flutter_spinkit.dart';
import 'package:go_router/go_router.dart';
import 'package:provider/provider.dart';
import 'package:security_app/providers/camera_provider.dart';
import 'package:security_app/widgets/error_widget.dart' as error_widget;
import 'package:security_app/widgets/empty_state_widget.dart';

/// Tab that displays the list of available cameras.
class CameraTab extends StatefulWidget {
  const CameraTab({super.key});

  @override
  State<CameraTab> createState() => _CameraTabState();
}

class _CameraTabState extends State<CameraTab> {

  @override
  void initState() {
    super.initState();
    // Defer fetch until after first frame to avoid calling provider during build.
    // Use read() instead of watch() because we don't need to rebuild on changes.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<CameraProvider>().fetchCameras();
    });
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

        return RefreshIndicator(
          onRefresh: () => cameraProvider.refreshCameras(),
          color: Theme.of(context).colorScheme.primary,
          strokeWidth: 3.0,
          backgroundColor: Colors.transparent,
          child: ListView.builder(
            itemCount: cameras.length,
            itemBuilder: (context, index) {
              final camera = cameras[index];

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
        );
      },
    );
  }
}