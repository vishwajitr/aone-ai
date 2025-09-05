#!/usr/bin/env python3
"""
Diagnostic script to check if all dependencies are properly installed
and your Angel One credentials are correctly configured.
"""

import sys
import importlib

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'smartapi',
        'pyotp', 
        'pandas',
        'numpy',
        'sklearn',
        'sqlite3',
        'asyncio',
        'threading',
        'logging',
        'json',
        'datetime'
    ]
    
    optional_packages = [
        'talib',
        'joblib'
    ]
    
    print("🔍 CHECKING REQUIRED DEPENDENCIES...")
    print("=" * 50)
    
    missing_required = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.import_module('sklearn.ensemble')
            else:
                importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError as e:
            print(f"❌ {package} - MISSING")
            missing_required.append(package)
    
    print("\n🎯 CHECKING OPTIONAL DEPENDENCIES...")
    print("=" * 50)
    
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"⚠️  {package} - OPTIONAL (system will use fallbacks)")
    
    if missing_required:
        print(f"\n❌ MISSING REQUIRED PACKAGES: {', '.join(missing_required)}")
        print("Install them with:")
        print(f"pip install {' '.join(missing_required)}")
        return False
    else:
        print("\n🎉 ALL REQUIRED DEPENDENCIES INSTALLED!")
        return True

def check_credentials():
    """Check if credentials are properly configured"""
    print("\n🔐 CHECKING CREDENTIALS...")
    print("=" * 50)
    
    # These are your current credentials from the code
    API_KEY = "FkpWlaE9"    
    CLIENT_CODE = "SPHOA1034"  
    PASSWORD = "0509"
    TOTP_SECRET = "7CWRXGUI2AB364N43NGHQNNHJY"
    
    issues = []
    
    if not API_KEY or len(API_KEY) < 5:
        issues.append("API_KEY looks incomplete")
    else:
        print(f"✅ API_KEY: {API_KEY[:4]}***")
    
    if not CLIENT_CODE or len(CLIENT_CODE) < 5:
        issues.append("CLIENT_CODE looks incomplete")
    else:
        print(f"✅ CLIENT_CODE: {CLIENT_CODE}")
    
    if not PASSWORD or len(PASSWORD) < 3:
        issues.append("PASSWORD looks incomplete")
    else:
        print(f"✅ PASSWORD: {'*' * len(PASSWORD)}")
    
    if not TOTP_SECRET or len(TOTP_SECRET) < 10:
        issues.append("TOTP_SECRET looks incomplete")
    else:
        print(f"✅ TOTP_SECRET: {TOTP_SECRET[:4]}***")
    
    if issues:
        print(f"\n⚠️  CREDENTIAL ISSUES: {', '.join(issues)}")
        return False
    else:
        print("\n🎉 CREDENTIALS LOOK GOOD!")
        return True

def test_totp_generation():
    """Test TOTP generation"""
    print("\n🔑 TESTING TOTP GENERATION...")
    print("=" * 50)
    
    try:
        import pyotp
        
        TOTP_SECRET = "7CWRXGUI2AB364N43NGHQNNHJY"
        totp = pyotp.TOTP(TOTP_SECRET)
        current_otp = totp.now()
        
        print(f"✅ TOTP Generated Successfully: {current_otp}")
        print(f"   (This should be a 6-digit number)")
        
        # Validate it's actually 6 digits
        if len(current_otp) == 6 and current_otp.isdigit():
            print("✅ TOTP format is correct")
            return True
        else:
            print("❌ TOTP format is incorrect")
            return False
            
    except Exception as e:
        print(f"❌ TOTP generation failed: {e}")
        return False

def test_smartapi_import():
    """Test SmartAPI import"""
    print("\n📡 TESTING SMARTAPI IMPORT...")
    print("=" * 50)
    
    try:
        from smartapi import SmartConnect
        print("✅ SmartConnect imported successfully")
        
        # Try to create instance (don't connect yet)
        api = SmartConnect(api_key="test")
        print("✅ SmartConnect instance created successfully")
        return True
        
    except Exception as e:
        print(f"❌ SmartAPI test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 AGENTIC OPTIONS TRADER - SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    results = []
    
    # Check dependencies
    deps_ok = check_dependencies()
    results.append(("Dependencies", deps_ok))
    
    # Check credentials
    creds_ok = check_credentials()
    results.append(("Credentials", creds_ok))
    
    # Test TOTP
    totp_ok = test_totp_generation()
    results.append(("TOTP Generation", totp_ok))
    
    # Test SmartAPI
    api_ok = test_smartapi_import()
    results.append(("SmartAPI Import", api_ok))
    
    # Final summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    all_good = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
        if not result:
            all_good = False
    
    print("\n" + "=" * 60)
    
    if all_good:
        print("🎉 SYSTEM READY! You can run the agentic trader now.")
        print("\nNext steps:")
        print("1. Make sure markets are open (9:15 AM - 3:30 PM IST)")
        print("2. Have sufficient margin in your Angel One account")
        print("3. Run: python agentic_options_trader.py")
    else:
        print("🔧 SYSTEM NEEDS FIXES! Address the failed items above.")
        print("\nCommon fixes:")
        print("1. Install missing packages: pip install pyotp smartapi-python")
        print("2. Verify your Angel One API credentials")
        print("3. Check your TOTP secret from Angel One app")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()