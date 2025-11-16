import 'package:security_app/models/camera_model.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

/// Service layer for camera-related API calls.
class CameraService {
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  final FirebaseAuth _auth = FirebaseAuth.instance;

  /// Fetches the list of cameras from Firestore that the current user
  /// is assigned to.
  Future<List<CameraModel>> getCameras() async {
    print("CameraService: Fetching cameras from Firestore...");

    final User? user = _auth.currentUser;
    if (user == null) {
      throw Exception('User not logged in. Cannot fetch cameras.');
    }
    
    try {
      // This query tests the Security Rule:
      // allow read: if request.auth.uid in resource.data.assignedUsers;
      final querySnapshot = await _firestore
          .collection('cameras')
          .where('assignedUsers', arrayContains: user.uid)
          .get();

      final cameras = querySnapshot.docs.map((doc) {
        return CameraModel.fromJson(doc.data(), doc.id);
      }).toList();

      print("CameraService: Found ${cameras.length} assigned cameras.");
      return cameras;

    } catch (e) {
      print("CameraService: Error fetching cameras: $e");
      // This error will trigger if the Security Rules fail
      throw Exception("Failed to load camera list: $e");
    }
  }

  /// Fetches the stream URL for a specific camera from its document.
  Future<String> getStreamUrl(String cameraId) async {
    print("CameraService: Getting stream URL for camera $cameraId");

    try {
      // This tests the 'get' part of the read rule
      final doc = await _firestore.collection('cameras').doc(cameraId).get();
      
      if (!doc.exists) {
        throw Exception("Camera not found");
      }
      
      final camera = CameraModel.fromJson(doc.data()!, doc.id);
      
      if (camera.streamUrl.isEmpty) {
        throw Exception("Stream URL is empty for this camera");
      }
      
      return camera.streamUrl;
      
    } catch (e) {
      print("CameraService: Error getting stream URL: $e");
      throw Exception("Failed to get stream: $e");
    }
  }
}