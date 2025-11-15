import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/settings_model.dart';

/// Provider for managing app settings with SharedPreferences persistence.
class SettingsProvider with ChangeNotifier {
  late SettingsModel _settings;
  bool _isLoading = true;
  String? _error;

  static const String _storageKey = 'app_settings';

  SettingsProvider() {
    _settings = SettingsModel.defaults();
  }

  // Getters
  SettingsModel get settings => _settings;
  bool get isLoading => _isLoading;
  String? get error => _error;

  // Settings properties
  bool get enableMotionAlerts => _settings.enableMotionAlerts;
  int get alertSensitivity => _settings.alertSensitivity;
  String get alertSound => _settings.alertSound;
  bool get darkMode => _settings.darkMode;
  int get refreshRateSeconds => _settings.refreshRateSeconds;
  bool get enableNotifications => _settings.enableNotifications;
  bool get showBadge => _settings.showBadge;

  /// Load settings from SharedPreferences
  Future<void> loadSettings() async {
    try {
      _isLoading = true;
      _error = null;
      notifyListeners();

      final prefs = await SharedPreferences.getInstance();
      final json = prefs.getString(_storageKey);

      if (json != null) {
        final decoded = jsonDecode(json) as Map<String, dynamic>;
        _settings = SettingsModel.fromJson(decoded);
        print('✅ Settings loaded from storage');
      } else {
        _settings = SettingsModel.defaults();
        print('📝 Using default settings (none saved yet)');
      }
    } catch (e) {
      _error = e.toString();
      _settings = SettingsModel.defaults();
      print('❌ Failed to load settings: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  /// Update a specific setting and persist to storage
  Future<void> updateSetting(String key, dynamic value) async {
    try {
      // Update in-memory settings
      switch (key) {
        case 'enableMotionAlerts':
          _settings = _settings.copyWith(enableMotionAlerts: value as bool);
          break;
        case 'alertSensitivity':
          _settings = _settings.copyWith(alertSensitivity: value as int);
          break;
        case 'alertSound':
          _settings = _settings.copyWith(alertSound: value as String);
          break;
        case 'darkMode':
          _settings = _settings.copyWith(darkMode: value as bool);
          break;
        case 'refreshRateSeconds':
          _settings = _settings.copyWith(refreshRateSeconds: value as int);
          break;
        case 'enableNotifications':
          _settings = _settings.copyWith(enableNotifications: value as bool);
          break;
        case 'showBadge':
          _settings = _settings.copyWith(showBadge: value as bool);
          break;
        default:
          return;
      }

      // Persist to storage
      await _persistSettings();
      print('💾 Setting updated: $key = $value');
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      print('❌ Failed to update setting: $e');
      notifyListeners();
    }
  }

  /// Update multiple settings at once
  Future<void> updateSettings(Map<String, dynamic> updates) async {
    try {
      for (final entry in updates.entries) {
        await updateSetting(entry.key, entry.value);
      }
    } catch (e) {
      _error = e.toString();
      print('❌ Failed to update multiple settings: $e');
      notifyListeners();
    }
  }

  /// Reset all settings to defaults
  Future<void> resetToDefaults() async {
    try {
      _settings = SettingsModel.defaults();
      await _persistSettings();
      print('🔄 Settings reset to defaults');
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      print('❌ Failed to reset settings: $e');
      notifyListeners();
    }
  }

  /// Clear all cached settings
  void clearCache() {
    _settings = SettingsModel.defaults();
    _isLoading = false;
    _error = null;
    notifyListeners();
  }

  /// Internal method to persist settings to SharedPreferences
  Future<void> _persistSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final json = jsonEncode(_settings.toJson());
    await prefs.setString(_storageKey, json);
  }
}
