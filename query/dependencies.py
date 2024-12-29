from fastapi import Depends


from .user import User

skip_auth = True

def validate_token():
    if skip_auth:
        return {"sub": "test", "permissions": ["test"]}
    
class AuthValidator:
    def __init__(self, required_permissions: list[str]):
        self.required_permissions = required_permissions

    def __call__(self, token: str = Depends(validate_token)) -> User:
        if skip_auth:
            return User(id="test-experts-51")
        if self.required_permissions.__len__() != 0:
            token_permissions = token.get("permissions")
            token_permissions_set = set(token_permissions)
            required_permissions_set = set(self.required_permissions)

            if not required_permissions_set.issubset(token_permissions_set):
                raise Exception
        return User(id=token.get("sub"))
