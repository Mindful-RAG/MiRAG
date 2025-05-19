{
  description = "MiRAG flake using uv2nix";

  inputs = {
    # pinning versions https://www.nixhub.io/
    # uv 0.6.3
    # python 3.12.8
    nixpkgs.url = "github:NixOS/nixpkgs/3a05eebede89661660945da1f151959900903b6a";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils = {
      url = "github:numtide/flake-utils";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:

      let
        inherit (nixpkgs) lib;

        workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

        # Create package overlay from workspace.
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel"; # or sourcePreference = "sdist";
        };

        pyprojectOverrides = _final: _prev: {
          # Implement build fixups here.
          # Note that uv2nix is _not_ using Nixpkgs buildPythonPackage.
          # It's using https://pyproject-nix.github.io/pyproject.nix/build.html
        };

        pkgs = import nixpkgs { inherit system overlay; };
        # pkgs = nixpkgs.legacyPackages.x86_64-linux;

        # Use Python 3.13 from nixpkgs
        python = pkgs.python312;

        # Construct package set
        pythonSet =
          # Use base package set from pyproject.nix builders
          (pkgs.callPackage pyproject-nix.build.packages {
            inherit python;
          }).overrideScope
            (
              lib.composeManyExtensions [
                pyproject-build-systems.overlays.default
                overlay
                pyprojectOverrides
              ]
            );

      in
      with pkgs;
      {

        apps = {
          default = {
            type = "app";
            program = "${self.packages.${system}.default}/bin/mirag";
          };
        };

        devShells = {
          impure = mkShell {
            packages =
              [
                python
                pkgs.uv
                pkgs.gcc
                pkgs.cmake
                pkgs.zlib
              ]
              ++ lib.optional (pkgs.stdenv.isLinux) [
                pkgs.rocmPackages.rocm-runtime # ROCm runtime
                pkgs.rocmPackages.rocm-smi # ROCm system management interface
                pkgs.rocmPackages.rocminfo
                pkgs.libdrm
              ];
            env =
              {
                # UV_PYTHON_DOWNLOADS = "never";
                # UV_PYTHON = python.interpreter;
                VIRTUAL_ENV = ".venv";
              }
              // lib.optionalAttrs pkgs.stdenv.isLinux {
                # Python libraries often load native shared objects using dlopen(3).
                # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
                LD_LIBRARY_PATH = lib.makeLibraryPath [
                  pkgs.pythonManylinuxPackages.manylinux1
                  pkgs.stdenv.cc.cc
                  pkgs.zstd
                  pkgs.zlib
                ];
                PYTORCH_ROCM_ARCH = "gfx1030"; # gfx1032
                HSA_OVERRIDE_GFX_VERSION = "10.3.0"; # 10.3.2
                PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True";
                HIP_VISIBLE_DEVICES = 0;
              };
            shellHook = ''
              # $SHELL
              # unset PYTHONPATH
              #
              uv sync --all-extras
              source .venv/bin/activate
            '';
          };

          uv2nix =
            let
              # Create an overlay enabling editable mode for all local dependencies.
              editableOverlay = workspace.mkEditablePyprojectOverlay {
                # Use environment variable
                root = "$REPO_ROOT";
                # Optional: Only enable editable for these packages
                # members = [ "hello-world" ];
              };

              # Override previous set with our overrideable overlay.
              editablePythonSet = pythonSet.overrideScope (
                lib.composeManyExtensions [
                  editableOverlay

                  # Apply fixups for building an editable package of your workspace packages
                  (final: prev: {
                    mirag = prev.mirag.overrideAttrs (old: {
                      # It's a good idea to filter the sources going into an editable build
                      # so the editable package doesn't have to be rebuilt on every change.
                      src = lib.fileset.toSource {
                        root = old.src;
                        fileset = lib.fileset.unions [
                          (old.src + "/pyproject.toml")
                          (old.src + "/README.md")
                          # (old.src + "/src/MiRAG/main.py")
                        ];
                      };

                      # Hatchling (our build system) has a dependency on the `editables` package when building editables.
                      #
                      # In normal Python flows this dependency is dynamically handled, and doesn't need to be explicitly declared.
                      # This behaviour is documented in PEP-660.
                      #
                      # With Nix the dependency needs to be explicitly declared.
                      nativeBuildInputs =
                        old.nativeBuildInputs
                        ++ final.resolveBuildSystem {
                          editables = [ ];
                        };
                    });

                  })
                ]
              );

              # Build virtual environment, with local packages being editable.
              #
              # Enable all optional dependencies for development.
              virtualenv = editablePythonSet.mkVirtualEnv "mirag-env" workspace.deps.all;

            in
            pkgs.mkShell {
              packages = [
                virtualenv
                pkgs.uv
              ];

              env = {
                # Don't create venv using uv
                UV_NO_SYNC = "1";

                # Force uv to use Python interpreter from venv
                UV_PYTHON = "${virtualenv}/bin/python";

                # Prevent uv from downloading managed Python's
                UV_PYTHON_DOWNLOADS = "never";
              };

              shellHook = ''
                # $SHELL
                # Undo dependency propagation by nixpkgs.
                unset PYTHONPATH

                # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                export REPO_ROOT=$(git rev-parse --show-toplevel)
              '';
            };
        };

      }
    );
}
