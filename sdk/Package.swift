// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "MiyraaSDK",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15),
        .tvOS(.v13),
        .watchOS(.v6)
    ],
    products: [
        .library(
            name: "MiyraaSDK",
            targets: ["MiyraaSDK"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MiyraaSDK",
            dependencies: [],
            path: "ios"
        ),
        .testTarget(
            name: "MiyraaSDKTests",
            dependencies: ["MiyraaSDK"]
        ),
    ]
)
