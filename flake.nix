{
  description = "Machine Learning environment based on Python and PyTorch.";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, flake-parts, ... }@inputs:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      perSystem = { self', inputs', system, ... }: let
        pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            (python311.withPackages (python-pkgs: with python-pkgs; [
              torch-bin
              torchvision-bin
              numpy
              pandas
              matplotlib
              pillow
              opencv4
              albumentations
              tqdm
            ]))

            python311Packages.black
            pyright
            ruff
            ruff-lsp
          ];
        };
      };
    };
}
